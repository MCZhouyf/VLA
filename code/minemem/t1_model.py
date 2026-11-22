"""Shared JARVIS-VLA policy and return-only LoRA objective."""

import torch
from torch import nn
from torch.nn import functional as F

from t1_spatial import SpatialMemory


HISTORY = {"no_memory": 2, "long": 8, "spatial": 2}
LORA_TARGETS = (
    r"model\.layers\.\d+\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))"
)


class T1Policy(nn.Module):
    def __init__(self, checkpoint, variant, quantized=False, adapter=None, training=False):
        super().__init__()
        from jarvisvla.inference.action_mapping import OneActionTokenizer
        from jarvisvla.inference.processor_wrapper import ProcessorWrapper
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

        if torch.cuda.device_count() != 1:
            raise RuntimeError("T1 policy requires exactly one allocated, visible GPU")
        self.variant = variant
        self.history = HISTORY[variant]
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.images = ProcessorWrapper(model_name="qwen2_vl")
        self.actions = OneActionTokenizer(tokenizer_type="qwen2_vl")
        kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                      local_files_only=True, device_map={"": 0})
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
        self.memory = SpatialMemory(dim, dim).to("cuda") if variant == "spatial" else None

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
        return self.processor(text=[template], images=frames, return_tensors="pt").to(self.device)

    def action_ids(self, action):
        ids = self.processor.tokenizer.encode(action, add_special_tokens=False)
        if not ids or ids[0] != self.actions.act_beg_id or ids[-1] != self.actions.act_end_id:
            raise ValueError(f"Invalid JARVIS-VLA action label {action!r}")
        return ids

    @torch.no_grad()
    def embed_images(self, pixel_values, image_grid_thw):
        self.base.visual.eval()
        return self.base.visual(pixel_values.to(self.base.visual.get_dtype()), grid_thw=image_grid_thw)

    @torch.no_grad()
    def pooled_images(self, frames):
        values = self.processor.image_processor(images=frames, return_tensors="pt").to(self.device)
        features = self.embed_images(values["pixel_values"], values["image_grid_thw"])
        merge_size = self.base.config.vision_config.spatial_merge_size
        counts = (values["image_grid_thw"].prod(dim=1) // merge_size**2).tolist()
        return torch.stack([tokens.float().mean(0) for tokens in features.split(counts)]).cpu().half()

    def context(self, batch, target=None, features=None, poses=None):
        ids = batch["input_ids"]
        if target is not None:
            ids = torch.cat((ids, target), dim=1)
        position_ids, _ = self.base.get_rope_index(ids, image_grid_thw=batch["image_grid_thw"])
        embeddings = self.base.get_input_embeddings()(ids)
        visual = self.embed_images(batch["pixel_values"], batch["image_grid_thw"])
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

    def loss(self, instruction, frames, previous_actions, action, features=None, poses=None):
        targets = torch.tensor([self.action_ids(action)], device=self.device)
        inputs = self.prompts(instruction, frames, previous_actions)
        embeddings, position_ids = self.context(inputs, targets, features, poses)
        hidden = self.base.model(inputs_embeds=embeddings, position_ids=position_ids,
                                 use_cache=False).last_hidden_state
        # Only the positions predicting the current return action enter the loss.
        logits = self.base.lm_head(hidden[:, -targets.shape[1] - 1:-1])
        return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())

    @torch.inference_mode()
    def predict(self, instruction, frames, previous_actions, features=None, poses=None):
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
        tags = self.actions.act_beg_id in ids and self.actions.act_end_id in ids
        return self.actions.decode(ids)[0], self.processor.tokenizer.decode(ids), tags
