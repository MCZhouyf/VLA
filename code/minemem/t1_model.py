"""Shared JARVIS-VLA policy and return-only LoRA objective."""

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from t1_spatial import SpatialMemory, pool_vision_grid


HISTORY = {"no_memory": 2, "long": 8, "spatial": 2}
LORA_TARGETS = (
    r"model\.layers\.\d+\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))"
)


class EncodedFrame(NamedTuple):
    grid: tuple[int, int, int]
    tokens: torch.Tensor


class T1Policy(nn.Module):
    def __init__(self, checkpoint, variant, quantized=False, adapter=None, training=False,
                 device="cuda", dtype=torch.bfloat16):
        super().__init__()
        from jarvisvla.inference.action_mapping import OneActionTokenizer
        from jarvisvla.inference.processor_wrapper import ProcessorWrapper
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

        if device == "cuda" and torch.cuda.device_count() != 1:
            raise RuntimeError("T1 policy requires exactly one allocated, visible GPU")
        if device == "cpu" and (training or quantized):
            raise ValueError("CPU T1 evaluation requires an unquantized checkpoint")
        self.variant = variant
        self.history = HISTORY[variant]
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.images = ProcessorWrapper(model_name="qwen2_vl")
        self.actions = OneActionTokenizer(tokenizer_type="qwen2_vl")
        kwargs = dict(torch_dtype=dtype, attn_implementation="sdpa",
                      local_files_only=True, device_map={"": 0 if device == "cuda" else device})
        if quantized:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"],
            )
        base = Qwen2VLForConditionalGeneration.from_pretrained(checkpoint, **kwargs)
        checkpoint_options = {"use_reentrant": False}
        if training:
            if quantized:
                base = prepare_model_for_kbit_training(
                    base, use_gradient_checkpointing=True,
                    gradient_checkpointing_kwargs=checkpoint_options,
                )
            else:
                base.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_options)
        if adapter:
            self.model = PeftModel.from_pretrained(base, adapter, is_trainable=training)
        else:
            self.model = get_peft_model(base, LoraConfig(
                r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
                target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
            ))
        self.base.visual.requires_grad_(False)
        dim = self.base.config.hidden_size
        self.memory = SpatialMemory(dim, dim).to(device) if variant == "spatial" else None

    @property
    def base(self):
        return self.model.get_base_model()

    @property
    def device(self):
        return self.base.get_input_embeddings().weight.device

    def prompts(self, instruction, frames, previous_actions):
        if len(frames) != len(previous_actions) + 1:
            raise ValueError("Expected one current frame and one action per history frame")
        messages = []
        for index in range(len(frames)):
            prompt = (instruction if index == 0 else "") + "\nobservation: \n"
            messages.append(dict(role="user", content=[dict(type="text", text=prompt),
                                                        dict(type="image")]))
            if index < len(previous_actions):
                messages.append(dict(role="assistant", content=previous_actions[index] + "\n"))
        template = self.processor.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True)
        if isinstance(frames[0], EncodedFrame):
            marker = self.processor.image_token
            parts = template.split(marker)
            if len(parts) != len(frames) + 1:
                raise ValueError("Chat template does not contain one image token per frame")
            expanded = [parts[0]]
            for frame, suffix in zip(frames, parts[1:], strict=True):
                expanded.extend((marker * len(frame.tokens), suffix))
            batch = self.processor.tokenizer(["".join(expanded)], return_tensors="pt").to(self.device)
            batch["image_grid_thw"] = torch.tensor([frame.grid for frame in frames], device=self.device)
            batch["encoded_frames"] = frames
            return batch
        return self.processor(text=[template], images=frames, return_tensors="pt").to(self.device)

    def action_ids(self, action):
        ids = self.processor.tokenizer.encode(action, add_special_tokens=False)
        if not ids or ids[0] != self.actions.act_beg_id or ids[-1] != self.actions.act_end_id:
            raise ValueError(f"Invalid JARVIS-VLA action label {action!r}")
        return ids

    @torch.no_grad()
    def embed_images(self, pixel_values, image_grid_thw):
        visual = self.base.visual
        visual.eval()
        pixels = pixel_values.to(visual.get_dtype())
        if len(image_grid_thw) <= 3:
            return visual(pixels, grid_thw=image_grid_thw)
        # Qwen2-VL vision attention is independent across the images. Its SDPA
        # implementation builds a dense cross-image mask when they are batched.
        image_sizes = image_grid_thw.prod(dim=1).tolist()
        sizes = [sum(image_sizes[start:start + 3]) for start in range(0, len(image_sizes), 3)]
        return torch.cat([
            visual(chunk, grid_thw=grid)
            for chunk, grid in zip(pixels.split(sizes), image_grid_thw.split(3), strict=True)
        ])

    @torch.no_grad()
    def encode_frames(self, frames, storage="cpu"):
        values = self.processor.image_processor(images=frames, return_tensors="pt").to(self.device)
        grid = values["image_grid_thw"]
        tokens = self.embed_images(values["pixel_values"], grid).to(storage)
        shapes = grid.tolist()
        merge_area = self.base.config.vision_config.spatial_merge_size ** 2
        counts = [time * height * width // merge_area for time, height, width in shapes]
        return [EncodedFrame(tuple(shape), chunk)
                for shape, chunk in zip(shapes, tokens.split(counts), strict=True)]

    @torch.no_grad()
    def pooled_images(self, frames):
        if not isinstance(frames[0], EncodedFrame):
            frames = self.encode_frames(frames, storage=self.device)
        return torch.stack([frame.tokens.float().mean(0) for frame in frames]).cpu().half()

    @torch.no_grad()
    def spatial_images(self, frames, side=2):
        if not isinstance(frames[0], EncodedFrame):
            frames = self.encode_frames(frames, storage=self.device)
        merge = self.base.config.vision_config.spatial_merge_size
        return torch.stack([pool_vision_grid(frame.tokens, frame.grid, merge, side)
                            for frame in frames]).cpu().half()

    def context(self, batch, target=None, features=None, poses=None):
        ids = batch["input_ids"]
        if target is not None:
            ids = torch.cat((ids, target), dim=1)
        position_ids, _ = self.base.get_rope_index(ids, image_grid_thw=batch["image_grid_thw"])
        embeddings = self.base.get_input_embeddings()(ids)
        cached = batch.get("encoded_frames")
        if cached is None:
            visual = self.embed_images(batch["pixel_values"], batch["image_grid_thw"])
        else:
            visual = torch.cat([frame.tokens.to(self.device) for frame in cached])
        mask = ids.eq(self.base.config.image_token_id)
        if int(mask.sum()) != len(visual):
            raise RuntimeError("Vision tokens do not align with the chat template")
        embeddings = embeddings.masked_scatter(mask[..., None], visual.to(embeddings.dtype))
        if self.memory is not None:
            if features is None or poses is None:
                raise ValueError("The spatial variant requires aligned visual features and poses")
            prefix = self.memory(features, poses).to(embeddings.dtype)
            embeddings = torch.cat((prefix, embeddings), dim=1)
            length = prefix.shape[1]
            prefix_positions = torch.arange(length, device=ids.device).view(1, 1, -1).expand(3, 1, -1)
            position_ids = torch.cat((prefix_positions, position_ids + length), dim=2)
        return embeddings, position_ids

    def context_batch(self, contexts):
        width = max(embeddings.shape[1] for embeddings, _ in contexts)
        padding = [width - embeddings.shape[1] for embeddings, _ in contexts]
        embeddings = torch.cat([F.pad(values, (0, 0, pad, 0))
                                for (values, _), pad in zip(contexts, padding, strict=True)])
        positions = torch.cat([F.pad(values, (pad, 0))
                               for (_, values), pad in zip(contexts, padding, strict=True)], dim=1)
        mask = (torch.arange(width, device=self.device)[None, :] >=
                torch.tensor(padding, device=self.device)[:, None])
        return embeddings, positions, mask

    def action_logits(self, instruction, frames, previous_actions, action, features=None, poses=None):
        targets = torch.tensor([self.action_ids(action)], device=self.device)
        inputs = self.prompts(instruction, frames, previous_actions)
        embeddings, position_ids = self.context(inputs, targets, features, poses)
        hidden = self.base.model(inputs_embeds=embeddings, position_ids=position_ids,
                                 use_cache=False).last_hidden_state
        # Only the positions predicting the current return action enter the loss.
        return self.base.lm_head(hidden[:, -targets.shape[1] - 1:-1]).float().flatten(0, 1), targets.flatten()

    def batched_action_logits(self, examples):
        contexts, targets = [], []
        for instruction, frames, previous_actions, action, features, poses in examples:
            target = torch.tensor([self.action_ids(action)], device=self.device)
            inputs = self.prompts(instruction, frames, previous_actions)
            contexts.append(self.context(inputs, target, features, poses))
            targets.append(target.flatten())
        embeddings, positions, mask = self.context_batch(contexts)
        hidden = self.base.model(inputs_embeds=embeddings, position_ids=positions,
                                 attention_mask=mask, use_cache=False).last_hidden_state
        lengths = [len(target) for target in targets]
        selected = torch.cat([row[-length-1:-1] for row, length in zip(hidden, lengths, strict=True)])
        return zip(self.base.lm_head(selected).float().split(lengths), targets, strict=True)

    def loss(self, *example):
        logits, targets = self.action_logits(*example)
        return F.cross_entropy(logits, targets)

    def complete_action(self, ids):
        from jarvisvla.inference.action_mapping import remap_control_token

        try:
            start = ids.index(self.actions.act_beg_id)
            end = ids.index(self.actions.act_end_id, start + 1)
        except ValueError:
            return False
        controls = [
            remap_control_token(token, use_num=True, tokenizer_type=self.actions.tokenizer_type)
            for token in ids[start + 1:end]
        ]
        count = len(self.actions.bases)
        if any(not 0 <= group < count or not 0 <= value < self.actions.bases[group]
               for group, value in controls):
            return False
        return {count - 2, count - 1} <= {group for group, _ in controls}

    @torch.inference_mode()
    def predict(self, instruction, frames, previous_actions, features=None, poses=None,
                *, return_ids=False):
        inputs = self.prompts(instruction, frames, previous_actions)
        embeddings, positions = self.context(inputs, features=features, poses=poses)
        output = self.base.model(inputs_embeds=embeddings, position_ids=positions, use_cache=True)
        next_position = int(positions.max()) + 1
        ids = []
        for _ in range(32):
            token = self.base.lm_head(output.last_hidden_state[:, -1]).argmax(-1).item()
            ids.append(token)
            if token == self.actions.act_end_id or token == self.processor.tokenizer.eos_token_id:
                break
            output = self.base.model(
                input_ids=torch.tensor([[token]], device=self.device),
                position_ids=torch.full((3, 1, 1), next_position, device=self.device, dtype=torch.long),
                past_key_values=output.past_key_values, use_cache=True,
            )
            next_position += 1
        return self.decode_prediction(ids, return_ids=return_ids)

    def decode_prediction(self, ids, *, return_ids=False):
        prediction = (self.actions.decode(ids)[0], self.processor.tokenizer.decode(ids),
                      self.complete_action(ids))
        return (*prediction, ids) if return_ids else prediction

    @torch.inference_mode()
    def predict_batch(self, examples, *, return_ids=False):
        contexts = [self.context(self.prompts(instruction, frames, actions), features=features, poses=poses)
                    for instruction, frames, actions, features, poses in examples]
        embeddings, positions, mask = self.context_batch(contexts)
        output = self.base.model(inputs_embeds=embeddings, position_ids=positions,
                                 attention_mask=mask, use_cache=True)
        next_positions = positions.amax(dim=(0, 2)).view(1, -1, 1).expand(3, -1, 1) + 1
        actions = [[] for _ in examples]
        finished = [False for _ in examples]
        end_tokens = (self.actions.act_end_id, self.processor.tokenizer.eos_token_id)
        for step in range(32):
            tokens = self.base.lm_head(output.last_hidden_state[:, -1]).argmax(-1, keepdim=True)
            for index, token in enumerate(tokens.flatten().tolist()):
                if not finished[index]:
                    actions[index].append(token)
                    finished[index] = token in end_tokens
            if all(finished) or step == 31:
                break
            mask = F.pad(mask, (0, 1), value=True)
            output = self.base.model(
                input_ids=tokens, position_ids=next_positions + step, attention_mask=mask,
                past_key_values=output.past_key_values, use_cache=True,
            )
        return [self.decode_prediction(ids, return_ids=return_ids) for ids in actions]
