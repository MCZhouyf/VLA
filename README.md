# VETA

Workspace for the VETA project.

## MineMem 阶段 0

阶段状态和实测预检记录见 [阶段 0 汇报](results/REPORT_phase0.md)。在分配了 GPU 6、7 的 A100-40GB 主机上，将 `WORK` 设为已分配的绝对路径。默认使用本机 17890 代理，也可沿用已配置的代理环境变量。

如果直连 BAAI 的 SSH 在握手时超时，本轮从当前终端使用本机 HTTP CONNECT 代理成功登录。目标别名使用该作业的 SSH 配置。

```bash
ssh -o 'ProxyCommand=nc -X connect -x 127.0.0.1:17890 %h %p' BAAI-Server-Bowen
```

```bash
export WORK=/path/to/allocated/work
export GPU_IDS=6,7
source env.sh
bash code/minemem/phase0.sh check
export CUDA_VISIBLE_DEVICES=
mkdir -p "$TMPDIR" "$MINESTUDIO_DIR"
conda create -y -p "$WORK/envs/mcvla" -c conda-forge python=3.10 pip openjdk=8
```

从本轮 BAAI 作业容器直接访问清华镜像和 HF Mirror 元数据成功。如果使用该容器的直连网络，在首次执行 `source env.sh` 前设置 `export MINEMEM_PROXY=`。

如果 conda 默认源在目标机器上不可达，使用清华镜像创建同一个前缀环境。

```bash
conda create -y -p "$WORK/envs/mcvla" --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  python=3.10 pip openjdk=8
```

将报告列出的固定提交解包到 `$WORK/code/MineStudio` 和 `$WORK/code/JarvisVLA`。阶段 0 使用 MineStudio 的依赖和 JARVIS-VLA 的源码解码器。两个上游分别固定了不同的 PyAV 版本，所以下列命令使用 MineStudio 的 PyAV，并通过 `PYTHONPATH` 读取 JARVIS-VLA 源码。

```bash
python="$WORK/envs/mcvla/bin/python"
"$python" -m pip install 'numpy==1.26.4' 'torch==2.5.1' \
  'transformers==4.49.0' 'accelerate==1.2.1' 'huggingface-hub==0.28.1' \
  -e "$WORK/code/MineStudio"
hf="$WORK/envs/mcvla/bin/huggingface-cli"
"$hf" download CraftJarvis/SimulatorEngine engine.zip \
  --revision 48d4809cfddc7e2b85295e8c39b3c5e8c6d46ae7 --local-dir "$MINESTUDIO_DIR"
unzip -n "$MINESTUDIO_DIR/engine.zip" -d "$MINESTUDIO_DIR"
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 \
  "$python" code/minemem/download_phase0_models.py
bash code/minemem/phase0.sh random
bash code/minemem/phase0.sh mcu
bash code/minemem/phase0.sh mcu --scene visible_stone
bash code/minemem/phase0.sh capacity
```

下载脚本固定三个 checkpoint 的发布版本，检查文件大小及大文件 SHA-256，结果写入 `$WORK/results/phase0_model_downloads.csv`。已有部分文件可用于续传。

每次运行新建 `$WORK/results/phase0-<mode>-*/`，输出实验 YAML、逐步 CSV、汇总 CSV 和 MP4。随机动作试跑屏蔽 CUDA，并验证 reset 命令和两种动作接口。MCU 试跑仅暴露第一张分配卡，使用上游 `mine/mine_stone` 原子任务及其挖石奖励判定。当前脚本用本地 Transformers 贪心生成，FPS 统计包含动作生成与环境 step，不含视频编码。

`--scene visible_stone` 使用位于初始视野内的石墙和已装备的石镐验证闭环。该场景的结果单列，正常场景仍使用自然生成的森林出生点和上游初始库存。

吞吐测试为每个实例启动独立的 CPU/Xvfb 环境并执行 100 步。所有实例完成 reset 后同时开始计时，输出 `throughput.csv` 和各档位的 `instances.csv`。默认档位为 1、4、8、16、32、64，也可在 `capacity` 后指定档位。

在 BAAI 项目目录运行以下命令可启动阶段 0 监控。它每 60 秒查询 GPU 6、7，将采样写入 `results/phase0_gpu_watch.csv`。监控有单实例锁。MineStudio 源码就绪后，监控安装阶段 0 依赖并执行 CPU 随机试跑。CPU 试跑通过、JARVIS-VLA 源码与完整 checkpoint 到位后，监控选择一张空闲卡运行一次 MCU。启动前再次检查该卡。空闲条件为零计算进程、0% 利用率和显存占用不超过 256 MiB。

```bash
export WORK=/share/project/zhoubowen/zhy/zyf
export GPU_IDS=6,7 CUDA_VISIBLE_DEVICES=6,7 MINEMEM_PROXY=
cd "$WORK"
nohup bash code/minemem/phase0.sh watch </dev/null >>results/phase0-gpu-watch.log 2>&1 &
```

## MineMem 阶段 1

以下命令复现阶段 1 的原始验收。2026-11-21 复核后的 T1 试点和主对比使用下一节的新版入口。命令从 BAAI 项目目录执行，沿用阶段 0 的引擎、Conda 环境和权重。[阶段 1 验收报告](results/REPORT_phase1.md)

```bash
export WORK=/share/project/zhoubowen/zhy/zyf
export GPU_IDS=6,7 MINEMEM_PROXY=
cd "$WORK"
source env.sh
export CUDA_VISIBLE_DEVICES=
envs/mcvla/bin/python -m venv --system-site-packages envs/mcvla-phase1-vllm
envs/mcvla-phase1-vllm/bin/python -m pip install \
  'vllm==0.7.3' 'torch==2.5.1' 'transformers==4.49.0' 'matplotlib==3.10.1'
```

公开 GUI 检查在 `$WORK/code/MineStudio-muyao-72cdab3` 使用 CraftJarvis MineStudio 的固定修订 `72cdab3c6cb5db829cb00f48507a91da37cea3ce`。从官方源码归档下载并校验 SHA-256 `d3534b02b65f9e016cbd0a707bfdbb080d9f1c63f30c0ffa733fee16cdd98cfb`，再以 `tar --strip-components=1` 解包。该分支支持官方评测脚本的装备干扰配置。T1 和 5 格导航使用阶段 0 的 MineStudio 版本。[官方源码归档](https://codeload.github.com/CraftJarvis/MineStudio/tar.gz/72cdab3c6cb5db829cb00f48507a91da37cea3ce)

```bash
patch -p1 -d "$WORK/code/MineStudio-muyao-72cdab3" \
  <code/minemem/minestudio_muyao_runtime.patch
```

```bash
export PYTHONPATH="$WORK/code/JarvisVLA"
envs/mcvla/bin/python code/minemem/audit_jarvis_actions.py results/phase1_jarvis_phase0_actions.csv \
  results/phase0-mcu-JjgkiRLZ/steps.csv results/phase0-mcu-gIMHeiv5/steps.csv

gate="$(mktemp -d "$WORK/results/phase1-jarvis-gate-XXXXXXXX")"
bash code/minemem/phase1_jarvis.sh serve 6 18606 >"$gate/server-6.log" 2>&1 &
bash code/minemem/phase1_jarvis.sh serve 7 18607 >"$gate/server-7.log" 2>&1 &
bash code/minemem/phase1_jarvis.sh batch "$gate" 18606 18607
envs/mcvla/bin/python code/minemem/phase1_metrics.py gate "$gate/summary.csv" "$gate"/*/episodes.csv
```

两项有论文单任务结果的公开 GUI 配置各跑 10 个种子，选择合成工作台和炼制铁锭。5 格金柱导航采用同一公开 7B checkpoint 和官方聊天推理封装。每个门槛试例有独立模拟器运行目录。汇总脚本要求每项 10 个规定种子齐全，GUI 任务与论文值的差距在 15 个百分点以内，导航达到 6/10。

```bash
bash code/minemem/phase1_teacher.sh 1 1
bash code/minemem/phase1_teacher.sh 32 50
```

教师脚本输出批次目录，每个工作进程复用一个模拟器。目录下的 CSV 和 JSON 记录世界种子、布景、返回起点、两侧绕障参考及实际步数。使用以下命令汇总四档验收结果。

```bash
teachers=/path/to/phase1-teacher-batch
envs/mcvla/bin/python code/minemem/phase1_metrics.py teacher \
  "$teachers/summary.csv" "$teachers"/worker-*/teacher.csv
```

完成 JARVIS-VLA 和教师门槛后，停止阶段 1 自己的单实例服务，再用 `phase1_jarvis.sh pair` 在两张卡上各启动两个 7B 服务。T1 运行器要求门槛的原始 CSV。正式档另外验证试点的全部逐回合记录。VPT-BC 附表使用 `CraftJarvis/MineStudio_VPT.bc_early_game_3x` 的固定修订 `e1f124c003fa322ae1e0a07465c090b421586b04`，目录为 `$WORK/models/MineStudio_VPT.bc_early_game_3x`。

```bash
HF_HUB_DISABLE_XET=1 envs/mcvla/bin/huggingface-cli download \
  CraftJarvis/MineStudio_VPT.bc_early_game_3x \
  --revision e1f124c003fa322ae1e0a07465c090b421586b04 \
  --local-dir "$WORK/models/MineStudio_VPT.bc_early_game_3x"
envs/mcvla/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

tokenizer = Path(snapshot_download("openai/clip-vit-base-patch16", revision="main",
                                   allow_patterns=["*.json", "merges.txt"]))
assert tokenizer.name == "57c216476eefef5ab752ec549e440a49ae4ae5f3"
PY
```

```bash
pilot="$(mktemp -d "$WORK/results/phase1-t1-pilot-XXXXXXXX")"
bash code/minemem/phase1_jarvis.sh pair 6 18616 18626 "$pilot/gpu-6" &
bash code/minemem/phase1_jarvis.sh pair 7 18617 18627 "$pilot/gpu-7" &
# 四个服务健康检查就绪后，分给各工作进程一个端口。
ports=(18616 18626 18617 18627)
for shard in 0 1 2 3; do
  bash code/minemem/phase1_jarvis.sh t1 "$pilot/worker-$shard" \
    --base-url "http://127.0.0.1:${ports[shard]}/v1" \
    --gate-results "$gate"/*/episodes.csv --teacher-dir "$teachers" \
    --seeds-file "$WORK/results/phase1_t1_seeds.json" --shard "$shard" --end 5 &
done
```

正式批次使用 `--end 30 --pilot-dir "$pilot"`。使用 `envs/mcvla-phase1-vllm/bin/python code/minemem/phase1_t1_report.py OUTPUT shard1.csv … --seeds SEED …` 汇总冻结的种子，生成主表、VPT 附表和 Wilson 95% 区间曲线。绘图时可设置 `MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"`。

## T1 随机终点朝向和阶段 3

2026-11-21 复核意见解除阶段 1 的旧模型门槛。公开权重用于零样本参照，主对比使用同一公开骨干和教师数据微调。新版将终点朝向均匀采样在金柱方位 ±60° 以外，保留停留 20 步、干扰柱和遮挡。[验证报告](results/REPORT_phase3.md)

~~~bash
export WORK=/share/project/zhoubowen/zhy/zyf GPU_IDS=6,7
cd "$WORK"
frozen="$WORK/results/phase1_t1_seeds.json"
acceptance="$(mktemp -d "$WORK/results/t1-v4-acceptance-XXXXXXXX")"
bash code/minemem/t1_cpu.sh acceptance "$acceptance" 32 --legacy \
  "$WORK/results/phase1-teacher-Rk9bGTdG" \
  "$WORK/results/phase1-teacher-terrain-retry-oHyVJIwe"
envs/mcvla/bin/python code/minemem/t1_v4_report.py "$acceptance/summary.csv" "$acceptance"
~~~

四档新版教师各需完成 50 个种子并达到 95%。30、60、100 格的新版 180° 转身直走成功率分别须低于 10%。训练集覆盖四档各 25,000 帧和 5–10 格可见导航 20,000 帧。JPEG 归档保存在服务器的 data/。索引检查数据划分的场景种子、世界种子与轨迹去重。

评分会先检查模拟器终止、角色存活及相邻控制步的位姿连续性，再计算水平距离。可用 acceptance 的 --recheck ARCHIVED_TEACHER 复用一批已完成的新版教师回合，并复跑两个朝向条件的捷径。

~~~bash
mkdir -p "$WORK/data"
data="$(mktemp -d "$WORK/data/t1-v4-XXXXXXXX")"
bash code/minemem/t1_cpu.sh collect "$data/train" 32 --split train
envs/mcvla/bin/python code/minemem/t1_data_index.py "$data/index-smoke" \
  "$data/train" --frozen "$frozen" --smoke
for split in validation test; do
  bash code/minemem/t1_cpu.sh collect "$data/$split" 4 --split "$split"
done
envs/mcvla/bin/python code/minemem/t1_data_index.py "$data/index-full" \
  "$data/train" "$data/validation" "$data/test" --frozen "$frozen"
~~~

零样本试点为四档各五个冻结种子。随机、文本 STEVE-1 和 VPT-BC 使用 CPU。JARVIS-VLA 使用阶段 1 双卡四实例入口，给四个并发评测进程各分配一个端口。将下例对五个试点种子分别执行。

~~~bash
seed=2026112100
bash code/minemem/t1_exec.sh pilot "$WORK/results/t1-cpu-$seed" \
  --acceptance "$acceptance" --seeds-file "$frozen" --seed "$seed" \
  --models random steve1 vpt
bash code/minemem/t1_exec.sh pilot "$WORK/results/t1-jarvis-$seed" \
  --acceptance "$acceptance" --seeds-file "$frozen" --seed "$seed" \
  --models jarvis7b --base-url http://127.0.0.1:18716/v1
envs/mcvla-phase1-vllm/bin/python code/minemem/t1_pilot.py report \
  "$WORK/results/t1-v4-pilot-report" --seeds-file "$frozen" --inputs \
  "$WORK"/results/t1-{cpu,jarvis}-*/cliff_T1.csv
~~~

阶段 3 环境继承 mcvla，使用 Transformers 4.49.0、Torch 2.5.1、PEFT 0.14.0 和可选的 bitsandbytes 0.45.3。冒烟训练使用 5,000 帧索引，记录优化步损失和动作解码，再用训练后的 LoRA 跑实机闭环。正式训练使用完整索引，三个变体采用同样的数据、步数、种子、量化选择和输入分辨率。默认 BF16，显存不足时通过 --quantized 启用 QLoRA。

~~~bash
source env.sh
envs/mcvla/bin/python -m venv --system-site-packages envs/mcvla-phase3
envs/mcvla-phase3/bin/python -m pip install peft==0.14.0 bitsandbytes==0.45.3
smoke="$WORK/results/phase3-smoke-no-memory"
bash code/minemem/t1_exec.sh train 6 --dataset "$data/index-smoke/dataset.jsonl" \
  --output "$smoke" --variant no_memory --epochs 1 --budget-seconds 3600
bash code/minemem/t1_exec.sh eval 6 "$WORK/results/phase3-smoke-closed-loop" \
  --training-run "$smoke" --mode smoke
~~~

正式变体名为 no_memory、long 和 spatial。用无记忆完整训练输出执行 eval --mode nav，通过 5 格可见导航的 8/10 门槛后，以 --mode formal --no-memory-gate 提供该批导航记录。正式评测还需提供 --acceptance、--seeds-file 和每进程的冻结 --seed。使用 t1_comparison.py 核查训练配置一致性，汇总三变体四档 × 30 个种子的 CSV 和 Wilson 95% 区间曲线。

## T1 死亡与重生复核

新版布景继续使用冻结的 T1 v4 场景、初始返回位姿、预算和种子。启动时在距金柱约 543 格处设置重生平台。每个回合的配置保存布景命令与重生点。受控步记录健康状态、死亡计数、位姿与跳变标记。死亡或单步水平跳变超过 2 格使回合失败。[复核报告](results/REPORT_t1_respawn.md)

~~~bash
export WORK=/share/project/zhoubowen/zhy/zyf GPU_IDS=6,7
cd "$WORK"
v4="$WORK/results/t1-v4-guarded-K6ARCMSV"
frozen="$WORK/results/phase1_t1_seeds.json"
acceptance="$(mktemp -d "$WORK/results/t1-protected-XXXXXXXX")"
bash code/minemem/t1_cpu.sh protected "$acceptance" 32 --fixtures "$v4"
envs/mcvla/bin/python code/minemem/t1_respawn_report.py acceptance \
  "$acceptance/summary.csv" "$acceptance"
~~~

按冻结试点名单为每个种子运行一次下列命令。可以用 --distances 将同一种子的不同距离分给独立进程，并给每个进程独立目录。

~~~bash
seed=2026112100
output="$WORK/results/t1-protected-pilot-$seed"
bash code/minemem/t1_exec.sh protected-pilot "$output" --fixtures "$acceptance" \
  --seeds-file "$frozen" --seed "$seed" --models random steve1 vpt
envs/mcvla-phase1-vllm/bin/python code/minemem/t1_respawn_report.py pilot \
  "$WORK/results/t1-protected-pilot-report" --seeds-file "$frozen" \
  --inputs "$WORK"/results/t1-protected-pilot-*/cliff_T1.csv
~~~

历史 VPT-BC 复跑采用归档的旧版布景和旧判定。健康与死亡记录作为观测，随逐步轨迹输出。下面以 100 格成功回合为例。

~~~bash
history="$WORK/results/t1-vpt-history-2026112104"
bash code/minemem/t1_exec.sh historical-pilot "$history" --fixtures "$v4" \
  --seeds-file "$frozen" --seed 2026112104 --distances 100 --models random steve1 vpt
envs/mcvla-phase1-vllm/bin/python code/minemem/t1_respawn_report.py trajectory \
  "$history/vpt-trajectory.svg" --config "$history/d100-seed2026112104.json" \
  --outbound "$history/d100-seed2026112104-outbound.csv" \
  --trace "$history/d100-seed2026112104-vpt-return.csv.gz"
~~~

绘制其他策略的回程时，通过 `--policy-label` 指定图中的策略名称。

GPU 实验只使用 6、7 号卡。三个变体各完成 5,000 帧冒烟及闭环后，使用 protected-pilot 运行公开 JARVIS-VLA。训练模型的可见导航和正式评测使用新版保护布景，正式评测的 --acceptance 指向上方复核的布景目录。

实际冒烟入口为 `t1_exec.sh train` 和 `t1_exec.sh eval --mode smoke`。对三个变体使用同一个冒烟索引，传入 `--variant no_memory`、`--variant long` 或 `--variant spatial`。每个变体训练一轮，进程预算为 3,600 秒。训练退出后用保存的训练目录作为 `--training-run`。视觉前向最多把三张图放在同一组，保留图像分辨率、完整历史和模型的注意力范围。闭环的 `decoded_steps` 统计含完整标签、合法控制码和两个相机轴的动作。[7B 实机记录](results/REPORT_phase3_gpu.md)

## T1 可见性与导航对照

`visibility` 在原冻结位姿上对 36 个朝向做有柱、无柱两次渲染。两次画面对齐后，用颜色掩码及像素差确认金块。教师首见统计使用同一路线的两次配对回放。扫视策略只接收 RGB 与自身步数，计分器另用位姿计算到达半径、死亡和跳变。

~~~bash
export WORK=/share/project/zhoubowen/zhy/zyf GPU_IDS=6,7
cd "$WORK"
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-XXXXXXXX)"
review="$(mktemp -d "$WORK/results/t1-review-XXXXXXXX")"
fixtures="$WORK/results/t1-respawn-tTWwXxTb/verified-acceptance"
seeds="$WORK/results/phase1_t1_seeds.json"
bash code/minemem/t1_cpu.sh visibility "$review/visibility" 12 \
  --fixtures "$fixtures" --seeds-file "$seeds"
envs/mcvla/bin/python -B code/minemem/t1_shortcut_audit.py summary \
  "$review/visibility_summary.json" --seeds-file "$seeds" \
  --inputs "$review"/visibility/worker-*/visibility.csv
bash code/minemem/t1_cpu.sh forward-control "$review/forward" 1
~~~

候选布景加 `--candidate`，在候选中验证金柱转角遮挡加 `--screened-pillar`。候选的 4 区块渲染距离只写入各自的实例配置。原布景复跑可用 `--case DISTANCE SEED` 只补缺失回合。SQLite 实例数据库可以通过上述环境变量置于本机临时文件系统，实验输出仍写入项目目录。

重采集使用 `t1_cpu.sh collect` 的 `--screened`。索引同样加 `--screened`，分别校验新版 T1 场景和沿用协议标签的可见导航场景。

~~~bash
data="$WORK/data/t1-screened-hzw7nYp7"
envs/mcvla/bin/python -B code/minemem/t1_data_index.py "$data/index-smoke" \
  "$data/train" --frozen "$seeds" --screened --smoke
envs/mcvla/bin/python -B code/minemem/t1_data_index.py "$data/index-full" \
  "$data/train" "$data/validation" "$data/test" --frozen "$seeds" --screened
~~~

验证按所有回程动作 token 加权，另保留整条动作完全匹配率。已保存的最后检查点可通过 `t1_validation.py` 独立验证。为早测两版导航，`t1_exec.sh eval-cpu` 支持 `--mode nav` 和 `--mode nav_offset`，同一 BF16 检查点在 CPU 以 FP32 执行。评测记录保存推理设备与精度。

~~~bash
envs/mcvla-phase1-vllm/bin/python -B code/minemem/t1_validation_plot.py \
  "$review/val_loss.svg" --full "$WORK/results/phase3-7b-0tyjJU3p/full"
~~~

[可见性、对照与训练监控复核](results/REPORT_phase3_followup.md)

## 第二版布景训练和恢复

正式训练使用第二版冻结索引。无记忆和空间记忆优先，两张卡分别为 6 和 7。长窗口排在两者之后。三个变体使用相同索引、分辨率、有效批大小和优化步数。每 1,000 个优化步验证整个 6,500 帧验证集，记录损失和 token 准确率，正式检查点取最后一步。

~~~bash
export WORK=/share/project/zhoubowen/zhy/zyf
cd "$WORK"
data="$WORK/data/t1-screened-hzw7nYp7/index-full/dataset.jsonl"
run="$(mktemp -d "$WORK/results/phase3-screened-XXXXXXXX")"
cache="$(mktemp -d /tmp/minemem-t1-archives-XXXXXXXX)"
df -h "$cache"
envs/mcvla-phase3/bin/python -B code/minemem/t1_stage_archives.py \
  --dataset "$data" --output "$cache" --manifest "$run/archive_sha256.jsonl"
bash code/minemem/t1_exec.sh train-resilient 6 "$run/train-no_memory" \
  --dataset "$data" --archive-cache "$cache" --variant no_memory --epochs 1 \
  --cache-vision --validation-batch-size 4
~~~

`train-resilient` 在同一张卡的独占锁内运行训练。如果进程异常退出，调度器从最近完整检查点恢复一次。检查点记录 LoRA、可选空间记忆、优化器、调度器、随机数状态和数据游标。训练保留最近两个快照。可以用 `t1_exec.sh train` 和原训练参数加 `--resume` 手动恢复。数据读取超时按 1、2、4、8、16 秒退避。训练从本地归档读取，冻结索引保持原样。归档准备程序记录每个 ZIP 的 SHA-256 并校验冻结索引中的动作与帧指纹。

~~~bash
CUDA_VISIBLE_DEVICES= PYTHONPATH="$WORK/code/JarvisVLA:$WORK/code/minemem" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 envs/mcvla-phase3/bin/python -B \
  code/minemem/test_t1_resume.py --checkpoint "$WORK/models/JarvisVLA-Qwen2-VL-7B"
~~~

`--cache-vision` 复用冻结视觉编码器输出，保持历史帧数和图像分辨率。`--validation-batch-size` 一次前向处理多个带独立掩码的样本。正式评测用 `t1_screened_fixtures.py --scores ACCEPTANCE_CSV --seeds-file SEEDS --output FIXTURES` 冻结已验收的布景，再将 `FIXTURES` 传给 `t1_exec.sh eval --mode formal --screened-fixtures`。用 `--no-memory-gate`、`--offset-gate`、`--spatial-gate`、`--spatial-offset-gate` 和 `--forward-control` 提供两组门槛与恒定前进记录。`--rollout-batch-size` 可在同一次模型生成中推进多个独立环境。

`t1_screened_eval.sh {gates|d30|remaining|all} VARIANT RUN FORWARD_CSV` 使用已冻结的场景与导航控制。`gates` 分别运行无记忆和空间记忆的两版导航。`d30` 使用 30 个固定种子，正式回程评测需要两个变体的门槛与结构对照通过。批量闭环的实际聚合吞吐写入 `rollout_timing.csv`。

新训练可在 `--accumulation 4` 下试用 `--train-microbatch-size 2`，把两条样本合并到一次前向与反向。每条样本仍按自己的动作 token 求平均，优化器仍每累计四条样本更新一次。默认微批大小为 1。使用同一设置启动各对照组，并先在空闲卡上测量吞吐和显存。空间记忆将仅供写入的早期外出帧先池化再复制到 CPU，并按格内时间顺序计算写入权重。

### GPU 6 和 GPU 7 运行监督

监督程序每 30 秒读取训练、验证、快照和显存状态，直到两条训练完成。对已确认的可恢复退出，如果原训练包装器尚未尝试恢复，监督程序核验冻结配置、归档清单和快照，在对应卡空闲时续跑一次。状态记录区分训练（含视觉编码）与验证，并跟踪评测等待器。持续异常与恢复决定写入事件日志。

~~~bash
run="$WORK/results/phase3-speed-xOn8Q5je"
mkdir -p "$run/gpu-supervisor"
nohup env CUDA_VISIBLE_DEVICES= "$WORK/envs/mcvla/bin/python" -B -u \
  "$WORK/code/minemem/t1_gpu_supervisor.py" --work "$WORK" --run "$run" --interval 30 \
  </dev/null >>"$run/gpu-supervisor/monitor.log" 2>&1 &
cat "$run/gpu-supervisor/latest.json"
tail -n 10 "$run/gpu-supervisor/events.jsonl"
~~~

[第二版启动与断点恢复验收](results/REPORT_phase3_screened_launch.md)

[GPU 6 无记忆训练与闭环评测报告](results/REPORT_phase3_no_memory_gpu6.md)

[阶段 3 第二版布景双卡训练与评测报告](results/REPORT_phase3_screened_training.md)

### 阶段 3 闭环控制复核

`diagnostic_d30` 模式在第二版冻结布景上只运行 30 格诊断，要求提供完整的 30 个冻结种子。正式评测需要无记忆和空间记忆的两版门槛 CSV，以及同场景的恒定前进记录。

~~~bash
export WORK=/share/project/zhoubowen/zhy/zyf
cd "$WORK"
original="$WORK/results/phase3-speed-xOn8Q5je"
run="$WORK/results/phase3-control-20261124-xeZzE4MM"

CUDA_VISIBLE_DEVICES= PYTHONPATH="$WORK/code/JarvisVLA" \
  envs/mcvla-phase3/bin/python -B code/minemem/t1_control_audit.py \
    --dataset "$WORK/data/t1-screened-hzw7nYp7/index-full/dataset.jsonl" \
    --checkpoint "$WORK/models/JarvisVLA-Qwen2-VL-7B" \
    --archive-cache "$(cat "$original/disk_cache_path")" --output "$run/action-audit-recheck.json"
~~~

诊断运行的 `spatial-gates.sh`、`diagnostic-d30.sh` 和 `gpu-control-probes.sh` 记录了实际命令。可见导航的 `--control-trace` 记录模型原始 token ID 和执行动作。`--start-probes` 记录初始输入逐 token 对比，并在同一画面测试原始、空、前进三种历史动作，以及 JPEG 输入对照。

~~~bash
bash code/minemem/t1_control_continue.sh data "$run" "$original"
bash code/minemem/t1_control_continue.sh train "$run" "$original" no_memory
bash code/minemem/t1_control_continue.sh train "$run" "$original" spatial
bash code/minemem/t1_control_continue.sh formal "$run" "$original"
~~~

`data` 用 32 路采集四档 T1 和可见导航的扰动轨迹，核验场景种子、世界种子、轨迹指纹及本地副本，冻结两来源各 6,000 条的继续训练索引。`train` 等对应诊断完成，从原末步 LoRA 和适用的空间模块初始化 3,000 步继续训练，写入动作变化子集的精确匹配率。`formal` 等两个变体的门槛结果齐全，记录门槛判定和允许启动时的 30 格比较结果。

如原权重诊断仍在 GPU 7 上运行，可用 `T1_SPATIAL_CONTINUATION_GPU=6` 启动空间记忆续训队列。该队列等无记忆导航完成后在 GPU 6 续训并评测导航，待 GPU 7 的原诊断结束才释放正式评测队列。两版导航和正式 30 格仍使用原定门槛。

服务器上的 `$run/control-repair-report.md` 每分钟根据落盘记录更新。 [首轮控制诊断报告](results/REPORT_phase3_control_diagnostics.md)

#### 偏角导航的平衡续训

使用首轮控制修复的冻结索引与本地已校验归档，从两组各自第 14,388 步权重重新初始化。每个优化步读取两条原始标签与两条 DART 标签，按同样顺序训练 1,000 步，每 250 步保存可恢复检查点。第 1,000 步验证并评测两版冻结导航。正式 30 格仍须两组都通过原定门槛。

~~~bash
export T1_CONTROL_MIX_RUN="$WORK/results/phase3-control-20261124-xeZzE4MM"
export T1_SPATIAL_CONTINUATION_GPU=7
balanced="$(mktemp -d "$WORK/results/phase3-steering-balanced-XXXXXXXX")"
for variant in no_memory spatial; do
  nohup bash code/minemem/t1_control_continue.sh train-balanced "$balanced" "$original" "$variant" \
    >"$balanced/continuation-$variant.log" 2>&1 </dev/null &
  printf '%s\n' "$!" >"$balanced/continuation-$variant.pid"
done
nohup bash code/minemem/t1_control_continue.sh formal "$balanced" "$original" \
  >"$balanced/repair-formal.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$balanced/repair-formal.pid"
~~~

平衡续训任一组导航未达标时，可令 `T1_STEERING_INIT_RUN="$balanced"`，新建输出目录并将上面的训练子命令换成 `train-weighted`。两组从各自的平衡续训末步适配器再训 750 步。每个更新用两条原始和两条 DART 标签，两来源均按索引比例均匀穿插可见导航样本。可见导航中有偏航的目标动作使用 3 倍相对权重，按批次权重和归一化动作交叉熵。

验证集的方向探针按直行、左转、右转等量抽样并记录自由生成。可在训练权重落盘后运行。

~~~bash
mix="$T1_CONTROL_MIX_RUN/mix"
CUDA_VISIBLE_DEVICES= PYTHONPATH="$WORK/code/JarvisVLA" \
  envs/mcvla-phase3/bin/python -B code/minemem/t1_steering_probe.py \
    --checkpoint "$WORK/models/JarvisVLA-Qwen2-VL-7B" \
    --training-run "$balanced/repaired/train-spatial" --dataset "$mix/dataset.jsonl" \
    --archive-cache "$(cat "$mix/archive_cache")" --device cpu --per-class 2 \
    --output "$balanced/spatial-steering-probe.json"
~~~

空间权重可加 `--compare-memory-read`，在相同输入和 LoRA 权重上记录保留与移除记忆前缀两种输出。这个选项用于单步诊断。
用 `--score-complete-actions` 可同时记录直行、最大左转、最大右转的完整序列概率，与原贪心输出对照。

配对转向采集在每个新位置记录目标位于正前方、左侧和右侧的三个场景。每个场景保留两帧静止历史和一帧教师动作。采集入口如下。

~~~bash
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-paired-XXXXXXXX)"
bash code/minemem/t1_cpu.sh paired-steering "$WORK/data/t1-paired-steering" 32 \
  --positions-per-worker 4
~~~

`t1_paired_steering.py freeze` 核验同位置、目标动作及与已有划分的场景和世界种子交集。`replay-index` 检查实际金柱像素方位，并补入原始与 DART 标签。完整运行的采集、数据冻结和 GPU 接续命令均在该轮结果目录的 `launch.sh` 与 `launch-train.sh` 中。`train-paired` 使用已校验的 `T1_PAIRED_MIX_RUN` 索引，从 `T1_STEERING_INIT_RUN` 的平衡续训权重以 0.00003 学习率训练两轮。每个优化步包含同位置的三条配对动作与三条回放动作。

#### 空间记忆的保留与利用

重放冻结轨迹可比较不同网格容量下起始观测的保留情况。位姿控制的输入为历史和当前位姿，金柱坐标用于评测。CPU 入口如下。

~~~bash
python3 -B code/minemem/t1_memory_audit.py results/t1-grid-retention \
  --fixtures results/phase3-speed-xOn8Q5je/screened_fixtures.json \
  --trace-roots "$WORK"/results/t1-review-A8vp1K1p/screened-acceptance*/ \
    "$WORK"/results/t1-review-A8vp1K1p/alternate-final-seed/ \
    "$WORK"/results/phase3-speed-xOn8Q5je/eval-no_memory-d30/

T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-pose-replay-XXXXXXXX)" \
  bash code/minemem/t1_cpu.sh pose-replay "$WORK/results/t1-pose-controls" 32 \
    --fixtures "$WORK/results/phase3-speed-xOn8Q5je/screened_fixtures.json" --distances 30 100
~~~

位姿对照可以加 `--terminal-tolerance 0.1` 进行减速末段控制。加 `--archived-outbound-index "$WORK/results/t1-grid-retention/episodes.csv"` 时，使用网格审计定位的外出轨迹及冻结终点，并逐条检查归档配置。审计时用绝对路径指定 trace 根目录，便于分片进程从各自目录读取。

`t1_memory_intervention.py` 在固定画面、当前位姿和 LoRA 权重下对比真实历史、打乱的历史视觉内容、清零的历史视觉内容、旋转的历史位姿与移除读取前缀。使用 `--wider-grid-side 64 --grid-only` 可以仅比较 64 格宽与 128 格宽的网格。网格扩容保留既有权重与相对坐标归一化。`--history-stride 16` 从历史每 16 帧选择一帧，保留完整的近期窗口，供离线读取诊断使用。此模式加 `--focus-landmark` 还可比较放大起始历史帧中金柱后的动作输出。

`t1_rgb_anchor_audit.py --dataset INDEX --archive-cache CACHE --output OUTPUT` 用已有第二版数据的验证与测试回合寻找 RGB 历史锚点，报告所选位置到金柱的距离，同时核查打乱帧与位姿的对应关系、跳过停留段和直接选择第一帧的结果。

~~~bash
T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-rgb-pose-XXXXXXXX)" \
  bash code/minemem/t1_cpu.sh rgb-pose-replay "$WORK/results/t1-rgb-pose-replay" 14 \
    --dataset "$WORK/data/t1-screened-hzw7nYp7/index-full/dataset.jsonl" \
    --archive-cache "$(cat "$WORK/results/phase3-speed-xOn8Q5je/disk_cache_path")"
~~~

此命令从归档中的回程场景和位姿恢复环境，分别跟随四种锚点对应的逆序历史路径，输出相同成功半径下的实机轨迹与逐回合指标。

空间组的独立训练可用 `t1_exec.sh train GPU --variant spatial --frozen-controller-run NO_MEMORY_RUN --initialize-from SPATIAL_RUN`。优化器只更新空间模块，仍从空间组初始化读写权重。`--position-value-scale 0.5` 将位姿嵌入也传入注意力的值。训练配置记录这两个选项，评测从相同配置恢复读取方式。

正式 30 格结果可以直接从归档的逐步位姿复算。 [阶段 3 结果与分析](results/REPORT_phase3_closeout.md)

~~~bash
python3 -B code/minemem/t1_phase3_closeout.py \
  --run results/phase3-steering-paired-2AGguG2e/shared-adapter-probe/d30 \
  --fixtures results/phase3-speed-xOn8Q5je/screened_fixtures.json
~~~

#### 带几何值的记忆读写与两项验收

[阶段 3e 实验报告](results/REPORT_phase3e.md)

`t1_memory_cache.py` 仅加载冻结的视觉编码器，缓存 2×2 或 4×4 视觉网格。`t1_memory_probe.py` 使用训练划分统计量做标准化，在 CPU 上训练小型读出头，自动记录训练前后 20 条验证轨迹的前缀敏感性以及航点和金柱的可解码准确率。`--grid-side 64` 对应 128 格的物理宽度。 [验收报告](results/REPORT_phase3e_memory_gate.md)

~~~bash
source "$WORK/env.sh"
export HF_HUB_OFFLINE=1 PYTHONPATH="$WORK/code/minemem"
run="$WORK/results/phase3e-memory-20261125-G8bBJisL"
attempt="$(mktemp -d "$run/recheck-XXXXXXXX")"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
  "$WORK/envs/mcvla-phase3/bin/python" -B code/minemem/t1_memory_probe.py \
    --cache "$run/cache" --output "$attempt/probe" \
    --legacy-weights "$WORK/results/phase3-steering-paired-2AGguG2e/shared-adapter-probe/spatial.safetensors"
~~~

两项验收通过后，使用 `t1_cpu.sh dual-enclosure OUTPUT WORKERS --distances 30 --pairs 1` 测试双围挡实机场景。`--capture --split STR` 采集分层对照所需的历史与末尾八帧。`t1_dual_probe.py cache` 在真值航点和端点匹配检查通过后缓存视觉，`train` 用前置验收的空间权重初始化。选择器评测使用 `--predictions` 与 `--fixtures` 复原相同场景，三组共享位姿航点控制器。

#### 正式 T1 v3 与航点控制器

正式协议使用 [阶段 3e 复核原文](research/phase3e_review.md)。`t1_cpu.sh phase3e` 记录出生点远距观察、同一固定种子下两候选的随机归属，并按进入金柱或干扰柱 3 格半径的先后评分。正式 CPU 验收对每档 30 个冻结种子跑教师和五个脚本基线。`t1_waypoint.py` 从当前教师路线投影沿折线前看 6 格，训练和推理共用 10° 与 1 格的文本量化。

~~~bash
run="$(mktemp -d "$WORK/results/phase3e-formal-XXXXXXXX")"
T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-phase3e-smoke-XXXXXXXX)" \
  bash code/minemem/t1_cpu.sh phase3e "$run/smoke" 4 --seeds 4
cp "$run/smoke/exit_status" "$run/smoke.status"
bash code/minemem/t1_phase3e_stage_a.sh "$run"
~~~

阶段 A 按顺序验收 CPU 布景、采集四档和可见导航并补 DART、核验本地训练归档、在 GPU 6 续训通过原近距门槛的 LoRA，最后在 GPU 6 和 7 运行四档真值航点闭环。训练与评测保留控制器输出的动作 token、航点提示和真实位姿。`oracle_gate.json` 记录四档各 30 场及 `>=80%` 判定。

真值航点评测在每张卡上复用同一世界的两档场景，先采集两条外出轨迹，再逐档重新布景和评分。GPU 锁被占用时调度等待并重试，训练异常恢复沿用最近有效检查点。

正式目标侧按场景朝向与外出访问顺序分层，训练目标侧用独立随机序列生成。若已完成同一组冻结种子的 30 个 CPU 分片，执行 `bash code/minemem/t1_phase3e_stage_a.sh "$run" --reuse-acceptance` 可从验收归并开始。运行目录需含 `smoke.status`、`smoke` 和 `acceptance`，归并验证四档和六种策略的完整覆盖及每个冻结种子的目标归属。

采集启动失败时，调度按组别检查各分片的配额及归档，仅用原种子复跑未完成分片。若先前的采集调度进程已经退出，执行 `bash code/minemem/t1_phase3e_stage_a.sh "$run" --resume-collection` 可接续已冻结的运行。

#### 航点转向振荡诊断

`t1_waypoint_stabilizer_audit.py` 在原始动作日志上定位实验性稳定器会介入的首步。该结果用于选择复测场景。实机诊断使用独立输出目录，记录改写前的模型动作与实际执行动作。稳定器检测连续 24 步交替原地转向，随后在航点对准时给出最多 4 步的短前进。

~~~bash
PYTHONPATH=code/minemem python3 -B -m unittest discover -s code/minemem \
  -p 'test_t1_waypoint_stabilizer.py'
python3 -B code/minemem/t1_waypoint_stabilizer_audit.py --run "$run"
diagnostic="$(mktemp -d "$WORK/results/phase3e-stabilizer-XXXXXXXX")"
bash code/minemem/t1_exec.sh waypoint-eval 7 "$diagnostic" \
  --training-run "$run/controller" --metrics "$run/frozen/metrics.json" \
  --fixtures "$run/frozen/fixtures.json" --distances 10 --batch 3 \
  --seeds 2086000000 2086000008 2086000011 --stabilize-turns
~~~

远档真值航点评测中断后，可用补跑入口保留已落盘的评分。运行目录需含修复版评测快照、原四份评测目录和 `original-run`。补跑使用 GPU 6 和 7，恢复两档缺失种子，待四档的冻结种子全部对齐后生成总判定。

~~~bash
recovery="$(mktemp -d "$recheck/recovery-XXXXXXXX")"
python3 -B -u code/minemem/t1_phase3e_recover_eval.py "$recheck" "$recovery"
~~~

四档真值航点验收通过后，阶段 B 先缓存三组共用的冻结视觉特征。对回程监督状态保留当前帧及前八帧，对外出段每八步取一帧。验证和测试使用既有留出划分的全部 T1 轨迹。

~~~bash
stage_b="$(mktemp -d "$WORK/results/phase3e-stage-b-XXXXXXXX")"
python3 -B code/minemem/t1_phase3e_cache.py plan --run "$run" \
  --oracle-gate "$recovery/recheck_oracle_gate.json" --output "$stage_b/cache-plan.json"
bash code/minemem/t1_phase3e_cache.sh "$stage_b"
~~~

缓存校验通过后，使用本地盘上按 SHA-256 核对的缓存训练航点预测器。三组共享返回监督状态、初始投影、输出头和损失。短窗口分别使用当前帧及前 2 或 8 帧，空间组使用当前帧与外出历史的 32 个空间读出。控制器保持冻结。

~~~bash
bash code/minemem/t1_exec.sh waypoint-heads 7 \
  --plan "$stage_b/cache-plan.json" --cache "$verified_local_cache" \
  --initial "$verified_memory_weights" --output "$stage_b/heads" --epochs 24
~~~

#### v3 孪生交换离线验收

验证和原始测试每档各用 30 个全新的场景，各 15 个左侧与右侧目标。测试再按同一组真实教师位姿渲染交换金柱和干扰柱的孪生场景。两次渲染逐采样点核对位置与朝向。最后八个外出帧和四个返程状态经遮挡检查后，在同一孪生对中共用其真实渲染。原始渲染的各帧校验值仍保留。共享控制器不参与离线生成。

~~~bash
swap="$(mktemp -d "$WORK/results/phase3e-visual-swap-XXXXXXXX")"
"$WORK/envs/mcvla/bin/python" -B code/minemem/t1_phase3e_pairs.py "$swap/plan.json" \
  --formal "$run" --oracle-gate "$recovery/recheck_oracle_gate.json"
bash code/minemem/t1_phase3e_pairs_collect.sh "$swap"
~~~

缓存用 `waypoint-pair-features`，把两卡分别设置为 `--shard 0` 和 `--shard 1`。使用阶段 B 已有训练清单和缓存，加入本次 `holdout-index.json`。同一池化版本使用独立的输出目录。读出在训练划分标准化，分别对冻结验证和孪生测试计算起点及随后三步的围挡分类、航点方向、孪生翻转率和置信区间。

~~~bash
bash code/minemem/t1_exec.sh waypoint-pair-features 6 \
  --train-plan "$stage_b/cache-plan.json" --old-cache "$stage_b/cache" \
  --holdout "$swap/holdout-index.json" --output "$swap/features-2" --shard 0
OMP_NUM_THREADS=6 "$WORK/envs/mcvla-phase3/bin/python" -B code/minemem/t1_phase3e_pair_probe.py train \
  --cache "$swap/features-2" --train-plan "$stage_b/cache-plan.json" \
  --initial "$verified_memory_weights" --output "$swap/fit-2"
OMP_NUM_THREADS=6 "$WORK/envs/mcvla-phase3/bin/python" -B code/minemem/t1_phase3e_pair_probe.py score \
  --cache "$swap/features-2" --holdout "$swap/holdout-index.json" \
  --weights "$swap/fit-2/room-heads.pt" --output "$swap/offline-2"
~~~

通过离线孪生门槛后，`t1_phase3e_stage_b_d30.py` 使用通过验收的围挡选择权重。三种模型以预测围挡为条件输出航点，使用同一个冻结 7B 控制器和稳定器。每场记录被改写的动作数。真值航点和真值位姿折返从原来的 30 个冻结种子逐场归并。

30 格五组评测的判定、逐场评分与轨迹审计见 [阶段 B 完成报告](results/REPORT_phase3e_phase_b.md)。使用 Python 3 从归档的动作与位姿记录复算三组学习策略的逐场轨迹指标：

```bash
python3 -B code/minemem/t1_phase3e_closed_audit.py results/phase3e-visual-swap-DB17NqMW/closed-2
```

#### 阶段 B 补充实验 CPU 检查

从阶段 B 归档复算两候选路线和上一轮的失败轨迹指标。使用一个新的输出文件名。

```bash
python3 -B code/minemem/t1_phase3e_grid_check.py \
  results/phase3e-visual-swap-DB17NqMW/closed-2 --output grid-check-recheck.json
```

在已准备的视觉缓存上重跑无锚点注意力和孪生评测。变量 `stage_b` 和 `swap` 分别指向阶段 B 缓存目录和孪生运行目录，`no_anchor` 指向新的输出目录。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$WORK/envs/mcvla-phase3/bin/python" -B code/minemem/t1_phase3e_no_anchor.py \
  --train-plan "$stage_b/cache-plan.json" --cache "$swap/features-2" \
  --weights "$swap/fit-2/room-heads.pt" --holdout "$swap/holdout-index.json" --output "$no_anchor"
```

方案 P 复用 `t1_exec.sh waypoint-predictions`。使用 `--variant spatial --grid-plan visual` 或 `--variant spatial --grid-plan shuffle_visual`，`--room-weights` 和 `--offline-gate` 指向无锚点运行的 `room-head.pt` 与 `offline_gate.json`。30 格两组各用原先冻结的 30 个种子。 [补充实验 CPU 报告](results/REPORT_phase3e_supplement_cpu.md)

可在 CPU 上直接查看参数。正式运行时，评测入口记录加载依赖的开始与完成时间。

```bash
python3 -B code/minemem/t1_phase3e_predicted_eval.py --help
```

#### 阶段 B 方案 P 两组正式闭环

将 `WORK` 设为服务器上的已分配项目目录。变量 `formal` 指向阶段 3e 冻结布景和控制器目录，`oracle_gate` 指向已通过的共享控制器判定，`no_anchor` 指向本轮离线读出目录，`plan_output` 是 `$WORK/results/` 下的新输出目录。

```bash
WORK="$WORK" CUDA_VISIBLE_DEVICES= "$WORK/envs/mcvla-phase3/bin/python" -B -u \
  code/minemem/t1_phase3e_grid_eval.py --formal "$formal" --oracle-gate "$oracle_gate" \
  --offline "$no_anchor" --output "$plan_output"
```

调度器在 GPU 6 跑原视觉组，在 GPU 7 跑打乱视觉组。进程异常退出时只重试缺失种子。两组各覆盖 30 个冻结种子后，复算动作记录并写入 `d30_decision.json`。

#### 方案 P 路径跟踪修复复测

在方案 P 的运行命令末尾加 `--tracking-repair`，使用新的 `plan_output` 目录。两组使用同一套路径跟踪与大偏角转向约束。逐场表的 `heading_guard_steps` 记录新转向约束的覆盖次数，`stabilizer_steps` 继续单独统计原稳定器。

从原方案 P 的外出轨迹复算碰撞代理检查：

```bash
python3 -B code/minemem/t1_phase3e_grid_check.py results/plan-p-d30-bfIpQlbn/closed \
  --tracking-repair --output tracking-check-recheck.json
```

[修复与复测报告](results/REPORT_phase3e_plan_p_tracking.md)

#### 方案 P 冻结确认实验

在 `ed550b0` 的归档检出上生成独立种子，使用安装了 MineStudio 实验依赖的 Python。变量 `frozen_checkout` 指向 Git 归档检出，`seed_output` 指向新目录。

```bash
python3 -B code/minemem/t1_phase3e_confirmation_cpu.py plan "$frozen_checkout" "$seed_output"
```

服务器运行目录 `confirmation_run` 含冻结提交导出的 `frozen-source`、单独的 `harness`、种子清单及 SHA-256 清单。设置服务器项目目录 `WORK`，在 CPU 验收通过后用原控制器目录 `controller`、无锚点读出目录 `no_anchor` 与原真值航点验收文件 `oracle_gate` 开始 GPU 两卡确认。

```bash
bash "$confirmation_run/harness/t1_phase3e_confirmation_cpu.sh" "$confirmation_run"
PYTHONPATH="$confirmation_run/frozen-source/code/minemem:$WORK/code/JarvisVLA" \
  "$WORK/envs/mcvla-phase3/bin/python" -B -u "$confirmation_run/harness/t1_phase3e_confirmation_run.py" \
  --run "$confirmation_run" --controller "$controller" --offline "$no_anchor" --oracle-gate "$oracle_gate"
```

[确认实验协议与报告](results/REPORT_phase3e_confirmation.md)

#### 修复布景后的独立确认

使用已归档旧确认结果的检出生成新版种子。用修复后的实验工具作为 `harness`，继续使用 `ed550b0` 导出的 `frozen-source`。CPU 验收收集六项基线。GPU 调度读取种子清单里的布景版本，并在验收通过后使用相同布景。

```bash
python3 -B code/minemem/t1_phase3e_confirmation_cpu.py plan "$inventory_checkout" "$seed_output" --repaired-scene
T1_CONFIRMATION_SCENE=repaired bash "$confirmation_run/harness/t1_phase3e_confirmation_cpu.sh" "$confirmation_run"
```

[布景根因、修复与验证](results/REPORT_phase3e_scene_repair.md)

#### 原 v3 主确认与布景泛化

主确认使用 `c09b0c31` 预注册的种子。`confirmation_run` 的 CPU 评分合并原始评分与唯一缺失的 60 格场景补跑。该场景使用 `--scene ground-repaired`，只修复残留砾石。`generalization_run` 使用已验收采集的第二版布景和种子清单。

```bash
PYTHONPATH="$confirmation_run/frozen-source/code/minemem:$WORK/code/JarvisVLA" \
  "$WORK/envs/mcvla-phase3/bin/python" -B -u "$confirmation_run/harness/t1_phase3e_confirmation_run.py" \
  --run "$confirmation_run" --controller "$controller" --offline "$no_anchor" --oracle-gate "$oracle_gate" \
  --generalization "$generalization_run"
```

调度器校验完整的 CPU 评分与场景清单，为新版协议记录独立的 `gpu-eligibility.json`。保留旧 CPU 判定，脚本实测用于并列表和配对差值区间。主确认按 30、10、60、100 格运行 A，再运行 30 格的 B、C。泛化另在第二版布景的 30 格档运行两组。

B2 的动作转换恢复记录保存在运行目录的 `recovery-harness`。已有运行的续跑入口是 `bash "$confirmation_run/recovery-harness/continue.sh" "$confirmation_run" --controller "$controller" --offline "$no_anchor" --oracle-gate "$oracle_gate" --generalization "$generalization_run"`。它等待原调度器退出，再从已记录的评分继续处理同一组种子。

#### 学习式目标格、航点和位姿误差

用包含确认结果的归档检出生成 30 格新种子。在 MineStudio 实验 Python 环境训练目标格和航点指针，使用现有阶段 B 训练缓存、孪生留出以及冻结的视觉投影。

```bash
python3 -B code/minemem/t1_phase3e_confirmation_cpu.py plan "$inventory_checkout" "$seed_output" --plan-l
python3 -B code/minemem/t1_phase3e_pointer_train.py --train-plan "$stage_b/cache-plan.json" \
  --train-cache "$stage_b/cache" --holdout "$swap/holdout-index.json" --holdout-cache "$swap/features-2" \
  --room-weights "$no_anchor/room-head.pt" --output "$pointer_run/fit"
```

服务器运行目录含 SHA-256 清单与冻结原 v3 源码。完成 CPU 布景验收后，Plan L 调度使用 `eval-harness/t1_phase3e_pointer_confirm.py`，指定 `--run`、`--controller` 和 `--room-weights`。位姿噪声调度使用 `harness/t1_phase3e_pose_sweep.py`，指定 `--run`、`--controller`、`--offline` 和 `--oracle-gate`。读取归档汇总生成论文曲线和 CSV。

使用同一位姿调度入口并加 `--odometry-calibration "$calibration"`，可在独立运行目录中评测动作积分。校准文件通过 `python3 -B code/minemem/t1_phase3e_odometry.py "$archived_actions" "$calibration"` 从已记录的平地动作和位移拟合。

```bash
python3 -B code/minemem/t1_phase3e_pose_plot.py "$confirmation_run/confirmation-decision.json" \
  "$pose_run/pose-results.json" "$plot_output/pose-error-curve"
```

[补充实验与投稿日期核对](results/REPORT_phase3e_followup.md)

## 研究交接总报告

完整阶段事实、证据边界、原始结果索引和接手复核命令见
[`research/EXPERIMENT_HANDOVER_2026-10-17.md`](research/EXPERIMENT_HANDOVER_2026-10-17.md)。
覆盖首轮Echo MVP、V2–V18、后续强规则/视觉/外部语义筛查及最新MuJoCo反馈闭环。
该文档区分新环境试次、旧数据分析和软件fixture，不将有限证据写成学习必要性或CVPR已通过。
交接时应同时保留`/home/zyf/veta_echo_mvp/`前期档案；只复制本仓库不能获得全部原始证据。

## Echo MVP 审计

官方源码冻结于 `0c9e53f60470d06642bcb9b847679e830543735d`，隔离副本位于
`/home/zyf/veta_echo_mvp/echo`。阶段报告与日志交付于
`/home/zyf/veta_echo_mvp/artifacts/`；本仓库保留审计脚本和报告副本。

首次准备命令（仅在目标目录不存在时执行，禁止覆盖已有实验）：

```bash
mkdir -p /home/zyf/veta_echo_mvp/artifacts
git clone --depth 1 https://github.com/CatworldLee/Echo.git /home/zyf/veta_echo_mvp/echo
```

重新执行只读审计及官方示例 CSD 离线检索：

```bash
bash scripts/reproduce_echo_audit.sh
```

此命令创建新的 `rechecks/<UTC时间>-<PID>/`，不会覆盖首轮证据、安装依赖、调用生成模型、
启动服务端、打开浏览器或修改已有世界。宿主机进程/网络审计需要在允许读取宿主机状态的环境运行。
源码缺失时仅执行会在静态导入阶段失败的入口诊断；它不算 Minecraft 任务试验。

如需指定独立输出目录：

```bash
python3 scripts/audit_echo_mvp.py --echo-dir /home/zyf/veta_echo_mvp/echo --output /home/zyf/veta_echo_mvp/rechecks/manual-audit
```

检查首轮交付证据（不会启动 Minecraft 或调用模型）：

```bash
python3 scripts/check_echo_evidence.py --artifacts echo_mvp/artifacts --echo-dir /home/zyf/veta_echo_mvp/echo --scripts-dir scripts
```

输出目录必须不存在。当前不得运行官方 `tasks/run_icl_flow.py` 的主程序：它会调用模型并自动执行新任务，
而当前发布缺少 Agent 运行模块。审计结果 `MVP_BLOCKED` 不构成负迁移证据。

恢复版单次真实试验在已有匹配 runtime、模型代理配置和 lifecycle hook 就绪后执行：

```bash
python3 scripts/recovery_v2/run_reconstructed.py --label source --task-path /home/zyf/veta_echo_mvp/artifacts/recovery_v2/source_task.json --task-id source_stone_to_cobblestone --generate-csd true
```

此命令使用新的独立 1.19 世界、端口 `25678`、Bot `VetaEchoV2`，从已停服并校验哈希的本次实验 baseline 复制世界。
所有模型请求经持久化 15 次调用硬上限代理；每次 Agent 最长 240 秒。判定读取 `success_before_clear.json` 的真实目标库存，
不以进程退出码作为任务成功。新证据写入 `/home/zyf/veta_echo_mvp/artifacts/recovery_v2/echo_trials/<run_id>/`；只清理本次创建的进程。
目标试次可传 `--condition treatment --generate-csd false`；`--no-vision` 显式关闭视觉，默认保留视觉。

## Echo runtime 恢复 v2

恢复工作副本位于 `/home/zyf/veta_echo_mvp/runtime_recovery/Echo-Reconstructed`，以 Mindcraft
`bed8a5d1516b44254ed39d501b79c647af215d3a` 接入冻结 Echo；交付保存在
`/home/zyf/veta_echo_mvp/artifacts/recovery_v2/`。结果名称为 Echo-Reconstructed，源码兼容和离线回归通过不代表完整 Echo 闭环成功。

重放来源核对、导出实际逐文件 provenance 与完整源码 diff，不启动 Agent 或修改 Git index：

```bash
python3 /home/zyf/veta_echo_mvp/runtime_recovery/inspect_selected.py --source-archive /home/zyf/veta_echo_mvp/runtime_recovery/mindcraft-bed8a5d.tar.gz
python3 scripts/recovery_v2/export_provenance.py
python3 scripts/recovery_v2/core_regression.py
python3 scripts/recovery_v2/check_budget_proxy.py
python3 scripts/recovery_v2/check_recovery_evidence.py --require-final
```

第一条比较冻结上游和 Echo 的虚拟叠加基线；第二条核对实际恢复目录，输出 `artifacts/recovery_v2/source_provenance.csv`
和 `runtime_recovery/reconstructed_changes.patch`。完整实验的重放入口为 `/home/zyf/veta_echo_mvp/artifacts/recovery_v2/reproduce.sh`，
运行前按其帮助选择只读校验或明确的真实试次；所有结论以保存的环境效果和模型调用证据为准。

本轮实际结果：恢复版 crafting smoke 成功，真实新 CSD 经归档续跑进入检索和 ICAL；生成计划未执行。
独立 `MECHANISM_TEST` 的隐藏 stone 掉落反事实通过；四条件 Echo 负迁移试次未执行，不声称负迁移成立。
全局 15 次模型调用硬上限已经耗尽，以下 `agent` 入口会拒绝继续，不会重置预算。

```bash
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v2/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v2/reproduce.sh mechanism
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v2/reproduce.sh native
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v2/reproduce.sh --help
```

来源清单最终刷新后，可用 `python3 scripts/recovery_v2/finalize_manifest.py` 从真实归档重建运行清单、
模型元数据哈希与所属进程清理核查。`reproduce.sh check` 只更新派生检查报告，不改原始轨迹或调用模型。

## VETA P0-Causal Transfer Pilot（recovery_v3）

新交付目录 `/home/zyf/veta_echo_mvp/artifacts/recovery_v3/` 保留 v2 全部历史；不修改论文或算法核心。
本轮完成真实 stone 成功源轨迹、独立规则复查、缓存木斧第一步失败验证，以及两世界原生 Agent 首请求的零网络截获。
稳定相机采集精确相等不代表图像已入模；源 CSD 和四条件试跑均受新增预算及固定目标 ICAL 调用边界门控。
不运行会覆写 recovery_v2 的旧入口。

```bash
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v3/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v3/reproduce.sh --help
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v3/reproduce.sh environment
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v3/reproduce.sh native-input
```

后两条仅新建本轮独立资源，模型调用始终为0；不会重置旧账本。`agent`/`model`/`causal-trials`入口在未授权时拒绝启动。
证据派生与报告命令（只针对v3，真实run目录替换为对应的新输出）：

```bash
python3 scripts/recovery_v3/observation_audit.py --run /home/zyf/veta_echo_mvp/artifacts/recovery_v3/environment_runs/20261015T025911355250Z
python3 scripts/recovery_v3/prepare_stage_reports.py --run /home/zyf/veta_echo_mvp/artifacts/recovery_v3/environment_runs/20261015T025911355250Z
python3 scripts/recovery_v3/finalize_report.py
python3 scripts/recovery_v3/validate.py
```

## VETA Recovery V4 — Stage 0

新交付为 `/home/zyf/veta_echo_mvp/artifacts/recovery_v4/`，状态 `WAITING_FOR_BUDGET_APPROVAL`。
12次是可申请上限而非授权；原账本保持不变。新闸门覆盖失败、并发、重启和视觉请求；未授权时不监听推理代理。
离线检查发现历史system反馈不兼容当前chat template，已在独立传输适配中保留文本并归一角色，不修改算法核心。
`ECHO_RECONSTRUCTED_FIXED_GOAL`只完成零调用的组件和早停测试，没有生成圆石CSD或N/M计划。

```bash
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v4/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v4/reproduce.sh counter-tests
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v4/reproduce.sh contract-tests
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v4/reproduce.sh --help
```

`counter-tests`只运行本地假HTTP工程fixture；`contract-tests`只做原模块格式检查和现有容器中的离线CPU tokenizer，
不加载模型做推理、不调用收费API、不启动Minecraft。`model-gateway`在明确授权/计费口径确认前拒绝启动。
不得自行把 `authorization.json` 的待确认模板当成批准；后续应记录真实用户批准证据，而非重置旧轮额度。

开发侧汇总与校验命令为 `python3 scripts/recovery_v4/audit.py --finalize`，只重建本轮派生报告和来源补丁。

### V4 授权后执行与内容门控

用户随后明确批准本地coder390（用户报告Qwen 27B）12次、输入8192/输出1600token、API现金上限0元；
不新增GPU租用，既有电费和折旧排除在API预算外。Stage0的等待报告保存在 `recovery_v4/stage0_snapshot/`。
本轮实际使用2次请求，由原CSD组件从v3真实轨迹和原始JPEG生成完整CSD，合计2537token。
proc主事件正确，但视觉把stone识别为slab、bedrock识别为stone，错误传播到Inter；
因此源CSD未通过可信门槛，当前为 `SOURCE_CSD_REJECTED_CONTENT_ERROR / BLOCKED_ENGINEERING`。
不手改、不重采样、不用余下10次绕过门控；没有N/M规划或四条件试次，不启动HAB10组。
已关闭本轮预算账户并仅清理自建代理，既有模型服务、官方核心、共享runtime和旧实验均保留。

```bash
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v4/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v4/reproduce.sh counter-tests
python3 scripts/recovery_v4/approved_report.py --finalize
```

以上均不调用模型；最后一条从既有真实证据重建当前派生报告。旧 `audit.py --finalize` 仅适用Stage0，
在授权后会拒绝，不能用它覆盖当前结果。`source-csd` 为一次性归档生成入口，已经执行后拒绝重采样；
`model-gateway` 在关闭状态拒绝监听。生成时的适配代码另存于 `approved_execution_sources/`，与新增关闭闸门区分。

## Recovery V5 — Verified Experience Transfer Pilot

交付 `/home/zyf/veta_echo_mvp/artifacts/recovery_v5/`。独立source-only模板将v3真实stone成功轨迹转换为
`DETERMINISTIC_VERIFIED_SOURCE`，不是重新运行或修补Echo模型CSD；未知视觉属性留空，每字段有日志provenance。
基线 `ECHO_RECONSTRUCTED_WITH_SHARED_VERIFIED_MEMORY`，规划协议 `ECHO_RECONSTRUCTED_FIXED_GOAL`，
执行方式 `OPEN_LOOP_PLAN_REPLAY`，模型观察为文本，不冒充Official Echo端到端或跨对象视觉HAB。

V4只读；新任务明确要求承接原12次授权剩余10次。V5累计账本原样保留旧调用1/2，新规划调用为3/4，
总计4/12，本轮新增2次、2534token，余8次不继续使用。失败/重试也计数，N/M各最多1次新规划，在线上限各2次。
原CSD/检索/ICAL九核心和204个runtime源文件不变，原样例分数精确回归。

N首挖cobblestone，M引用认证源并首挖stone。原始两计划都用`pick_up`而非`pickup`，
首次严格schema失败完整保留；双方共用的词法映射在生成后、真实执行前冻结，仅改操作名，不改选择或顺序。
四条件各一次真实执行且最终都成功，只有T-M首次stone无掉落并增加一次dig；主成本Delta_Shift=+1。
这是认证记忆/缓存计划重放的受控单例描述，不是总体负迁移率、原生Echo自主闭环或VETA方法已验证。
全部自建4个Minecraft服务、4个Bot与推理代理已清理，既有coder390服务和旧世界保留。

```bash
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v5/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v5/reproduce.sh source-check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v5/reproduce.sh counter-tests
python3 scripts/recovery_v5/report.py --finalize
```

这些命令不调用模型、不重跑真实试次。`plan`/`execution`为已使用后拒绝的单次入口，不能重置或覆盖。
首次构建步骤为 `python3 scripts/recovery_v5/prepare.py` → `python3 scripts/recovery_v5/run_planning.py` →
`python3 scripts/recovery_v5/plan_format_adapter.py` → `python3 scripts/recovery_v5/run_execution.py`，
均保留明确的存在性/预算门控；当前已有产物，不能重复执行来追加采样。

## Recovery V6 — Closed-Loop Transfer Confirmation

交付 `/home/zyf/veta_echo_mvp/artifacts/recovery_v6/`，模式 `FROZEN_INIT_ONLINE_FEEDBACK`，输入 `TEXT_ONLY_CLOSED_LOOP`。
冻结V5同一源CSD及N/M初始actions，所有缓存recovery剔除；不启动原生Agent.start，不声称官方端到端。
薄接口每步调用原Task.validator检查真实库存并用原History存储环境反馈，需要时通过原Prompter.promptConvo实际调用模型。
没有stone特判或预设替代路径。双方同schema、拾取词法适配、控制器、目标判据和2次恢复机会。
原Task.isDone成功会清库存，故只调用其同一个原validator，不执行清库存/killAll副作用。

四组各一次且全部成功；T-M一次stone无掉落后，基于实时反馈在线调用模型一次，自主选择cobblestone并成功。
Total Dig Count为C-N=1/C-M=1/T-N=1/T-M=2，Delta_Shift=+1；有首次错误成本，也有有效在线恢复。
仅单例、认证记忆、冻结初始计划、文本反馈，不能外推显著性、跨对象视觉泛化或VETA方法已验证。
原始在线响应的未执行备用段仍含旧air坐标，因此不声称旧经验已永久修正。

本轮承接4/12后仅新增1次（3554输入+75输出=3629token），累计5/12、余7次封存不自动追加；API现金0元。
四个自建Java/Bot会话及代理均退出，旧coder390与V2–V5产物保持不变。圆石案例已冻结为regression case，不再围绕本例迭代。

```bash
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v6/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v6/reproduce.sh counter-tests
python3 scripts/recovery_v6/report.py --finalize
```

以上不会新增模型调用或世界；最后一条仅从真实日志重建本轮派生报告。
首次构建为 `python3 scripts/recovery_v6/prepare.py` → `node scripts/recovery_v6/test_components.mjs` →
`python3 scripts/recovery_v6/run_batch.py`。这些入口已有一次性存在性、原额度累计和封闭账户闸门；当前不能重复运行来追加样本。

## Recovery V7 — Cross-Object Transfer Pilot

交付 `/home/zyf/veta_echo_mvp/artifacts/recovery_v7/`。本轮没有新增模型授权，历史5/12余7不得自动使用；
V7硬上限0请求，没有启动推理代理、没有新初始计划或Agent四条件试次。复用冻结V5 Verified-CSD，
九核心、共享runtime及V2–V6原始证据不变，不重新生成经验或重跑旧圆石案例。

一次Control和一次Treatment确定性真实验证：源stone与目标infested_stone原始ID不同，C/T都为infested_stone；
仅Control目标loot datapack不同，目标入库C+1/T+0，双方替代路径各+1。源/目标与独立C/T初始RGB逐字节一致。
然而两组Peaceful下均实际出现蠹虫，违反预注册零蠹虫门槛，并触发既有实体渲染器`THREE is not defined`。
因此候选严格标为`CANDIDATE_INFEASIBLE`，没有事后放宽、换对象或追加采样；不能宣称跨对象负迁移成立。

新增白名单别名边界在真实观察与离线反泄漏测试通过，但正式Agent发送链路未运行。
原五维检索以冻结官方样例精确回归；新masked query仅做零模型诊断，检索不等于采纳。
当前真实RGB只做本地CPU图像processor审计，**没有发送给模型**。所有自建Java/Bot/Xvfb已退出，既有coder390保留。

```bash
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v7/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v7/reproduce.sh boundary-tests
```

以上命令不启动模型或Minecraft，不改变旧证据。首次零调用构建命令为
`python3 scripts/recovery_v7/prepare.py` → `python3 scripts/recovery_v7/run_mechanism.py` →
`python3 scripts/recovery_v7/audit.py`；前两项已有一次性门控，拒绝覆盖/重采样。
`audit.py`只重建本轮派生报告，首次可调用Docker内本地离线processor分词，不做模型forward。
本候选模型阶段及10组HAB均为NO-GO；若继续，须先重新预注册可行性/观察门槛、修复隔离渲染问题并另行确认预算。

## Recovery V8-A — Clean Cross-Object Feasibility

交付 `/home/zyf/veta_echo_mvp/artifacts/recovery_v8a/`，唯一新候选为Stone→Andesite，结束V7的infested_stone候选。
只做确定性机制、视觉、白名单和真实请求构造器的本地传输检查；新增LLM硬上限0，历史仍5/12，余7未授权。
冻结V5源CSD、九核心和共享runtime不变，不规划、不跑Agent四条件、不训练/下载视觉模型或扩展HAB。

独立Minecraft1.19、seed68422026、localhost25684、同木镐无附魔；仅Control的andesite loot表改成cobblestone。
预先固定出生点和spawnRadius0，并用`item replace`初始化工具，避免V7的临时工具实体。
真实目标C得到cobblestone×1、T得到andesite×1且无cobblestone；两组替代各得到cobblestone×1，
每组dig2、工具总磨损2，无蠹虫/其他额外实体/THREE或控制器错误。全部实体生成/metadata/消失及警告原样留存。

C/T初始独立RGB、玩家/库存/工具/公开文本完全匹配。源Stone与目标Andesite有98001/409600像素差异，
没有事先可靠冻结视觉编码器或语义阈值，故`SOURCE_TARGET_VISUAL=UNVERIFIED`，不能按灰色外观或全帧低误差硬判PASS。
实际未改`VLLM.sendRequest`/SDK已将两份相同、预遮蔽、含真实inline JPEG的请求送到私有本地假接收端，
仅2次mock HTTP、0次模型推理；`VISION_INPUT=READY_NOT_CONSUMED`不等于模型已消费RGB或原生Agent已多模态化。
机制/静态/名称均PASS，但总体`CANDIDATE_NOT_READY`，不自动进入跨对象四条件实验。

```bash
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v8a/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v8a/reproduce.sh boundary-tests
```

以上命令不运行Minecraft或模型、不覆盖旧证据。首次构建命令为
`python3 scripts/recovery_v8a/prepare.py` → `python3 scripts/recovery_v8a/run_environment.py` →
`python3 scripts/recovery_v8a/input_preflight.py` → `python3 scripts/recovery_v8a/report.py`。
前三步已有一次性门控，拒绝覆盖/重采样；第三步只允许自建localhost假接收端且不转发。
最后一步仅从现存真实记录重建本轮派生报告。所有自己创建的Java/Bot/Xvfb和假接收端已清理，原服务/世界/V2–V7证据保持不变。

## Recovery V8-B — Renderer-Level Visual Identity

交付 `/home/zyf/veta_echo_mvp/artifacts/recovery_v8b/`，新条件为`HAB_SYNTHETIC_VISUAL_IDENTITY`。
不追认V8-A原生Stone/Andesite相似性，不改冻结源CSD、共享runtime/viewer/atlas或九核心。
预注册相机/AABB射线ROI后，在本轮独立Camera字节副本的薄子类中，使用已有`WorldRenderer.blockStatesData`接口，
仅将andesite模型的材质和面UV指向原stone纹理；模型几何/物理/掉落不变，无atlas重建或依赖升级。

新独立Source场景只采图不挖掘；C/T复用V8-A原世界/loot快照并各做一次真实机制回归。
Source/Target在预冻结102294像素ROI及完整800×512帧中差异均0；C/T原PNG/JPEG、别名文本和实际SDK请求一致。
C目标掉圆石，T掉安山岩，双方替代各圆石+1，无生物/实体渲染异常。
最初因旧Server端口替换逻辑发生3次零帧/零动作的连接失败，完整日志保留在`infrastructure_failure_01/`；
仅修正自有模板端口后取得各一个有效场景，没有删除失败、重采样成功试次或放宽判据。

实际原VLLM/SDK向私有假接收端发送2次含本轮真实RGB的请求；新增LLM请求0，历史仍5/12、余7未授权。
结果为`CANDIDATE_READY_FOR_VISUAL_AGENT_GATE`、`VISION_PIPELINE=READY_NOT_CONSUMED`，不是模型消费、
四条件结果、HAB验证或负迁移证据。当前安全观察接口不开放会返回原始块名的native中心视线工具，
也不把原CSD文本直接拼入请求；下一Agent Gate仍需强制安全工具/记忆边界、视觉消费验证与明确硬预算。

```bash
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v8b/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v8b/reproduce.sh boundary-tests
```

以上只复核现存证据，不启动世界或模型。首次执行链路为
`python3 scripts/recovery_v8b/prepare.py` → `python3 scripts/recovery_v8b/install_renderer.py` →
`python3 scripts/recovery_v8b/run_environment.py` → `python3 scripts/recovery_v8b/input_preflight.py` →
`python3 scripts/recovery_v8b/report.py`。前四步有存在性/零额度门控；本轮实际用过一次
`run_environment.py --repair-port-before-any-physics`，仅在既存失败全部为连接拒绝且没有动作/有效帧时允许一次，
保留失败日志并规范化自有端口；现在该修复和运行入口均拒绝重用。所有自建资源已清理，原服务与V2–V8-A产物不变。

## Recovery V8-C — Stage0 Zero-Inference Preflight（已冻结）

交付 `/home/zyf/veta_echo_mvp/artifacts/recovery_v8c/`，当前`WAITING_FOR_AUTHORIZATION`，
任务分类`PILOT_PARTIAL_OR_BLOCKED`。本轮新增LLM请求/模型接收图像均0，没有新世界、Agent、gateway或诊断场景。
历史仍5/12、未用7；该余额不构成V8-C授权，当前硬cap0。所有V5–V8-B及更早档案、九核心、共享runtime、CSD/renderer patch不变。

实际完成归档逐文件/源CSD provenance校验、原五维精确回归、固定目标与在线入口审计、
既有真实图像请求的CPU本地processor分词，以及8项零网络guard单测（含24个并发未授权保留探测，均拒绝且不改真实账本）。
旧V6 raw feedback/history与native中心块名/nearbyBlocks/inventory工具不能直接开放；
原SDK包没有token硬限制，必须经过未来逐请求安全边界/硬gateway。原源记忆上下文含stone名称，
与先前Source/Target raw-ID仅评测端的准则需要明确冻结公共记忆视图；当前不改CSD或prompt，也不宣称发送集成已完成。

保守完整预算提案为视觉2+规划2+四组各1恢复=最多8新请求、累计13，须显式将原12 cap增加1；
这只是提案，未激活。若只批准诊断/规划最多4次，可留在原cap内，但之后余3不能启动需四组各预留一次恢复的匹配执行。
每次输入≤8192、输出≤1600，失败重试计数、SDK retries0；本地API现金0/无付费外部API或新增GPU租用，费用口径仍须按明确授权执行。

```bash
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v8c/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v8c/reproduce.sh guard-tests
```

以上不启动模型或世界。首次命令为`python3 scripts/recovery_v8c/audit.py --initialize`，已存在时拒绝覆盖；
`audit.py`无参数仅从既有记录重建Stage0报告，首次会在原容器用CPU离线processor分词，不执行model forward。
当前没有真实推理执行入口，`request_guard.py`仅提供关闭授权闸门与组件检查，不可把单元测试通过视为在线Agent安全或视觉消费通过。

## Recovery V8-C — Approved Vision/Planning Gates

用户后续明确批准最多4次，仅视觉诊断与初始规划，原费用/token上限沿用；Stage0原文件与源码保存在
`artifacts/recovery_v8c/stage0_frozen/`，当前报告以授权后结果为准。没有增加原12请求总额度或授权四条件。

在任何新模型响应前冻结两张独立真实红/蓝场景、真值/中性问题/判据，以及新N/M完整输入；
实际coder390分别回答`red/pixelated`、`blue/pixelated`，通过`VISUAL_BEHAVIOR_PASS`，不是只检查接口200。
N/M均收到同一真实Andesite（渲染为Stone）RGB；N首挖替代圆石[-1,64,0]，M引用冻结源CSD并首挖目标[1,64,0]。
原M输出`pick_up`保留，沿用预冻结双方共享的词法映射，不改动作选择。计划层采纳和分离通过，**没有执行计划或成本证据**。

原CSD、五维检索和fixed-goal指令不变；检索后仅将公开记忆context的历史stone标识别名为source_block，
当前世界仍target_block，投影在任何模型请求前冻结。原始CSD/hash和原/公开context均留档，不冒充Official Echo。
原SDK经独立持久计数gateway逐请求检查实际文本/图像/冻结payload与含图输入token，SDK retries0、失败计数。
本轮4/4请求、4224token（3764输入+460输出），模型接收4份图像消息、3张唯一图；累计9/12，原余3，当前授权余0。
API现金0元，无新增GPU租用/外部付费API。3个自建拍摄世界/Bot/Xvfb及gateway已清理，原服务和V2–V8-B证据不变。

当前完整Pilot分类`PILOT_PARTIAL_OR_BLOCKED`只因Stage3未获批且未运行，不是视觉/规划Gate失败。
`03_trials.jsonl/03_metrics.csv`未创建，不算Delta，不宣称经验失效、负迁移、闭环恢复或HAB验证。
原余3不足给四组各预留1次在线恢复；不能擅自扩预算或启动有利的一半，也不自动启动HAB10组。

```bash
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v8c/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v8c/reproduce.sh guard-tests
```

以上不新增模型或世界，当前guard-tests是隔离active-account工程测试并验证已关闭真实账户拒绝重开。
首次批准后执行链：`authorize.py` → `capture_scenes.py` → `freeze_inputs.py` → `test_active_budget.py` →
`run_approved.py` → `approved_report.py`（均位于`scripts/recovery_v8c/`，Python执行）。
前几步和模型batch已有一次性存在性门控，账本关闭后不能借原余3重开；`approved_report.py`仅重建派生报告。
旧`audit.py`在授权后禁止覆盖Stage0，仅`--check`委托当前完整性校验。完整源码来源/差异与实际请求/图hash/响应/usage均归档。

## Recovery V8-C — Four Conditions and Audited Checkpoint Continuation

后续用户授权继续完成后，实际运行四条件各一次；没有训练VETA、扩大对象或启动HAB10。
本轮工程自限Stage3最多4次、每组1次、累计13，沿用8192/1600 token、API现金0元，
不是把“先不设限制”当无限Agent。实际Stage3只发请求10一次，模型接收新的真实RGB，
input3258/output75，按真实反馈自行选择替代圆石并完成任务；原核心/CSD/检索/renderer保持不变。

严格结论仍`PILOT_PARTIAL_OR_BLOCKED`，不能把下面继续结果叫干净四条件PASS：
原T-M恢复入口将历史V6同名`trial_02`误计为本轮已用，尚未调用模型便提前停止，原批次3成功/1工程中断。
所有原始轨迹和执行源码保留在`stage3_interrupted_frozen/`及`stage3_frozen_scripts/`。
只从原T-M保存世界/库存/History继续，没有重放初始actions或产生新episode；
两个零动作/零请求启动问题（既存RGB目录、yaw相差2pi的等价表示）分别存档并修复。
`03_checkpoint_amendment.json`在新响应前冻结，明确这是工程协议偏差，不抹去原失败。

继续后真实库存四组均达标；T-M共2dig/1无效dig/磨损2，其他三组各1dig/0无效/磨损1。
继续段描述性`Delta_T=1, Delta_C=0, Delta_Shift=1`，不是无偏干净主因果估计或显著性证明。
原`03_trials.jsonl`/`03_metrics.csv`和附加`03_continued_trials.jsonl`/`03_metrics_continued.csv`分开；
不把工程暂停计为经验负迁移，不把坐标/程序复制单独归因为视觉类比。全部自有资源退出，原模型PID/镜像不变。
V8-C总请求5、图像消息5（4张唯一JPEG），历史累计10；原12未用2、工程cap13未用3，已关闭可花费0。

```bash
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v8c/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v8c/reproduce.sh guard-tests
```

上述仅校验现有证据和隔离工程测试，不创建世界或推理请求。实际首次执行命令（现在均不可重用）为：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v8c/prepare_stage3.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v8c/test_stage3.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v8c/run_stage3.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v8c/continue_checkpoint.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v8c/continue_checkpoint.py --repair-preaction-mkdir
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v8c/continue_checkpoint.py --repair-preaction-yaw
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v8c/stage3_report.py
```

最后一个仅重建派生报告，不重新执行试次；原Stage0/Stage1/2报告禁止覆盖最终阶段。
原始/继续/失败启动各有源码hash与生命周期；缺失的强制退出Xvfb cleanup事件如实披露，以实际PID不存在核验退出。
当前NO-GO进入HAB10；下一干净设计需新版本明确预注册，不允许自动重跑此冻结候选来制造PASS。

## Recovery V8-C2 — Clean Engineering Repair Validation

用户在“独立新批次、保留旧结果、相同冻结计划/规则、各条件一次”的修复方案后明确回复继续。
新交付目录为`/home/zyf/veta_echo_mvp/artifacts/recovery_v8c2/`，与旧V8-C完全隔离；
旧V8-C仍保留`PILOT_PARTIAL_OR_BLOCKED`，不覆盖失败、续跑或账本。

新批次实际完成T-N/T-M/C-N/C-M各一次，**无中断、无检查点续跑、无重采样，四组真实库存均成功**。
新状态`CROSS_OBJECT_PILOT_PASS`仅针对固定既有计划的工程修复验证：沿用已选V8-C请求8/9的N/M计划，
不是新的独立规划样本或显著性证明。九个Echo核心、源CSD、五维检索、原renderer/loot/世界模板均未改。

主指标dig为T-M=2，其余三组均1，`Delta_T=1, Delta_C=0, Delta_Shift=1`；
T-M有1次无效目标效果、磨损2，其余磨损1；新真实RGB反馈触发模型自主选择替代圆石并成功。
T-M仅执行1个恢复动作（dig后已拾得目标），T-M和T-N总环境动作均4，不能说所有成本增加。
本批次只有1次新模型请求（id11），真实新图像消息1份，3261输入/53输出/3314总tokens。
历史累计11，原12未用1、工程累计cap14未用3，账户已关闭可花费0；现金API0元，无付费外部API或新增GPU租用。

修复包括按experiment+phase+unit隔离计数、独立run id、幂等RGB目录以及禁止检查点的硬入口；
原型修复后的runtime只增加隔离目录/端口/task日志ID等薄适配，没有改经验、算法或策略。
服务端端口25688，私有gateway18093，模型仍原localhost8000服务；Node使用原已验证的
`/home/zyf/.local/opt/node-v22.23.3-linux-x64/bin/node`，不修改全局环境。
完整执行源码在任何新世界/模型响应前冻结并从归档运行，报告源码另存，所有差异/provenance及真实payload/帧/usage留档。

```bash
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v8c2/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v8c2/reproduce.sh guard-tests
```

上述只校验现有证据与隔离工程测试，不启动模型或世界。实际首次执行命令为：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v8c2/prepare.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v8c2/tests.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v8c2/run.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v8c2/report.py
```

prepare/run已关闭且一次性，不得重用；report仅重建派生报告。新四个Java/Bot/Xvfb及gateway正常退出，
原coder390 PID/镜像/config不变，旧V2–V8-C全部文件哈希保护通过。
工程可进入另行预注册的小规模下一阶段；HAB10仅条件性GO，需要先冻结样本选择、统计单位、
视觉/坐标/程序复用的归因边界及硬预算，本轮不自动启动。仍不训练VETA、不扩对象、不改论文。

## Recovery V9 — Stage A–C Zero-Inference Preparation

交付目录`/home/zyf/veta_echo_mvp/artifacts/recovery_v9/`，状态`WAITING_FOR_MODEL_AUTHORIZATION`。
本附件重新要求独立模型授权，过去“继续”或旧账户余额不自动适用；本轮新增模型请求/收图/推理tokens均0。
最新历史实际11次（包含独立V8-C2请求11），不是附件引用的V8-C历史10次；所有旧产物及状态保持不变。

根因已用原始源码/JSONL/账本实际复现：V8-C frozen controller第216行按phase+unit过滤，
误把V6请求5计入V8-C trial_02，第217行普通break提前结束；不是Mineflayer字段变化或dig重复事件。
该主中断没有异常栈；检查点启动EEXIST的完整真实栈另列，yaw异常只有原type/message，不能编造丢失堆栈。
复用已存在的命名空间修复，V9显式区分experiment；新增被动计数器，不进入模型History、不决定动作，监测异常不传播。
原Total Dig Count仍是受预算检查后的digAttempts++，不偷偷改成只计成功dig；另外区分调用/完成/破坏/预期掉落/库存增量/无效效果。

实际完成10项原真实轨迹回放测试，以及2项独立Minecraft确定性工程回归：
Case1目标正常掉圆石，1dig；Case2首挖掉Andesite、目标圆石增量0，再按预定替代资源完成，2dig/1无效效果。
两项都由原库存validator及server真值核对，renderer初始JPEG/PNG、安全文本均匹配，所有自建进程正常退出。
它们不是Echo Agent试次，不生成CSD、不进入四条件主表，不将确定性绕行冒称模型自主恢复。

Stage C已冻结新run id、C-N→T-M→T-N→C-M顺序、原CSD/模板/规则/目标/计划/Prompt/mask/renderer、
统一4dig/16动作/1恢复/240秒/4800ticks与完整执行源码；禁止检查点或第二轮尝试。
Stage E没有执行，没有05_trial_manifest/05_trials/05_metrics，也不计算V9 Delta或声称INTEGRITY_PASS。
只做模型GET健康/元数据核查，未发送推理POST；32个并发未授权保留探测及正式入口均拒绝且账本不变。

```bash
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v9/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v9/reproduce.sh counter-tests
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v9/reproduce.sh guard-tests
```

以上不启动Minecraft或推理；回放结果内容确定且不修改历史文件。实际首次执行链为：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v9/prepare.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v9/tests.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v9/run_regression.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v9/freeze_protocol.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v9/report.py
```

prepare/环境回归/协议冻结均有一次性存在性门控，report仅重建A–C派生报告。
正式`run_formal.py`在新V9授权前、创建世界/gateway之前硬拒绝；不要尝试借旧余额调用。
预算提案V9-PROPOSAL-ONLINE-004尚未批准：最多4次本地coder390在线恢复，每组最多1次，
输入8192/输出1600，失败计数、SDK retries0、现金API0、无外部收费API/新增GPU租用；若批准最大累计15。
旧12上限未用1、V8-C2工程cap14未用3都不等于V9授权；当前V9活动额度0。
不自动启动HAB10，不训练/开发VETA，不扩候选/基线，不修改论文。

## Recovery V9 — Approved Prospective Four-Condition Run

用户在独立4次预算提案后回复“继续”，授权记录为`V9-AUTH-ONLINE-001`，绑定原冻结协议。
Stage A–C原报告、cap0账本及源码完整存档于`artifacts/recovery_v9/stage_abc_frozen/`；
四条件的执行器、参数、CSD、Prompt、规则、纹理、mask与初始计划全部保持原Stage C字节不变。

实际从归档源码按C-N→T-M→T-N→C-M各执行一次，四个新隔离世界均真实库存成功，
**无工程中断、无检查点、无补跑**，最终`INTEGRITY_PASS`及`GO_TO_HAB_PILOT_DESIGN`。
Total Dig为C-N1/C-M1/T-N1/T-M2，`Delta_T=1, Delta_C=0, Delta_Shift=1`；
T-M实际破坏Andesite但无目标圆石效果，经新RGB和安全反馈请求得到模型新计划，再自行开采替代资源并拾取成功。
T-M共5个环境动作/2恢复动作/磨损2，其余磨损1；完整原始计数见`05_metrics.csv`，不混入Stage B或旧版本。

本轮仅新增1次模型请求（全局id12），模型收到1份真实新图像消息，3265输入/59输出/3324总tokens。
历史累计12，独立授权4次尚未用3，四组结束后账户关闭不可自动消费。API现金0元，无外部付费API/新增GPU租用。
48项并发预算fixture只使用隔离临时账本、不发送网络；真实生产账本保留全部前11条且没有复位。
所有自有Java/Bot/Xvfb正常退出、gateway关闭、原模型PID/镜像/config不变；所有旧实验保持hash一致。

```bash
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v9/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v9/reproduce.sh counter-tests
bash /home/zyf/veta_echo_mvp/artifacts/recovery_v9/reproduce.sh guard-tests
```

上述校验不启动世界或推理。授权后的首次实际命令为：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v9/authorize.py
PYTHONDONTWRITEBYTECODE=1 python3 /home/zyf/veta_echo_mvp/artifacts/recovery_v9/stage_c_frozen_scripts/run_formal.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recovery_v9/final_report.py
```

前两步均一次性，不能重开或补跑；最后一步只重建派生报告。原A–C `report.py`在授权后禁止覆盖最终报告，
guard-tests改为安全隔离fixture，不可运行旧未授权guard来探测活动真实账本。
离线报告曾因注册manifest缺少时间字段而报KeyError，已使用真实run_started与文件mtime核验，不伪造注册时间；
未影响已完成的四条件或改动执行源码。完整说明在`06_integrity_audit.md`。
本轮仅是固定既有计划的单实例前瞻性执行复核，不是独立新规划样本、统计显著性或VETA方法优势。
GO仅进入另行预注册HAB10设计，不自动运行10组，不混算旧版本成功率，不训练f_phi、不改论文。

## Recovery V10 — HAB-10 Design And Pre-registration

本轮仅离线设计，源码在`scripts/hab10_design/`，交付到
`/home/zyf/veta_echo_mvp/artifacts/hab10_design/`；不调用模型、不启动Minecraft或正式Agent。
10个不同源—目标模板，SIMILAR/DIFFERENT各5个，配对形成20机制情境/40计划运行；
Stone→Andesite仅为V9开发回归，排除于HAB独立样本。实际认证源目前只有Stone一条，
新的Cobblestone源轨迹/CSD待采，明确null、不伪造第二份认证；目标10种ID。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_design/register.py build
bash /home/zyf/veta_echo_mvp/artifacts/hab10_design/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/hab10_design/reproduce.sh tests
```

`build`只允许新目录，发布后拒绝覆盖。`check`只读核验全V9、九核心及冻结配置，并重生成比对。
`tests`在临时目录做配置变异/泄漏/确定性测试，不运行环境或网络。
单独重生成到一个尚不存在的新scratch目录：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_design/generate.py \
  --spec scripts/hab10_design/spec.json \
  --registry /home/zyf/veta_echo_mvp/artifacts/hab10_design/registry_snapshot.json \
  --output /tmp/hab10_NEW_scratch_directory
```

这是配置而非世界快照/真实观测。renderer沿用V8-B的纹理/UV覆写接口；V9的mask有固定坐标和对象名，
新布局需要后续位置感知薄适配、实拍RGB/安全文本等价及确定性机制验收，不改九核心或历史CSD。
未来预算仅提案：20个新初始计划（C/T共享）+最多40个在线恢复=60次本地请求，
8192输入/1600输出、失败计数/SDK重试0、API现金0；历史累计12及已关闭余额不能使用。
设计就绪不等于执行就绪；S2认证、机制/视觉/遮蔽门、新协议冻结和新显式预算授权前不得启动40组。

最终设计成本复核使用V9原始时间戳重算生命周期跨度76.381728秒，替代初稿未经核实的94.56秒。
初版完整保存在`artifacts/hab10_design/audit_revisions/r0_before_final_cost_audit/`，
修订理由/hash在`design_review_revision.json`；候选、规则、布局、预算和历史实验均不改变。
本轮一次性封存命令为`PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_design/register.py seal-review`；
封存后该命令拒绝再次修订。常规只使用`check`和`tests`，不覆盖预注册。

## V11 — HAB-10 Execution Readiness

源码`scripts/hab10_readiness/`，数据写入
`/home/zyf/veta_echo_mvp/artifacts/hab10_readiness/`。只认证S2及10模板的20个真实机制情境，
不调用LLM、不开放V10的60次提案、不运行40次Agent，不改V9/V10或九核心。

本轮实际一次性命令链：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_readiness/audit.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_readiness/run_environment.py source
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_readiness/certify.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_readiness/transport.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_readiness/run_environment.py H01
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_readiness/run_environment.py H06
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_readiness/validate_cases.py canary
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_readiness/batch.py
```

S2使用V5原始`build_csd`与同一广义证据验证（S1精确回归），唯一新经验ID为002，
原子事实、物品实体、服务器库存真值、轨迹和RGB可追溯；不是原生MLLM自动经验。
只在新`bound_candidate_manifest.json`绑定S2，不重生成/改写V10的候选。

首批H03_C在任何dig前因初始渲染地平线327像素时序差异失败并立即停批；
原失败、六个先前成功case、旧canary与源码保留在原路径/`engineering_revisions/capture_v1/`。
隔离采集层只新增固定2500ms初始等待和500ms mesh/work静默门，不按像素挑帧或放宽0差异标准，
不改纹理、对象、规则、布局、工具；按共同capture_v2重做全部20情境验收，而非只补一个有利case：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_readiness/reaccept.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_readiness/interface.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_readiness/report.py
```

这些live入口均为一次性，不能覆盖现场；仅在对应真实阶段门通过时执行。正式Agent入口不存在。
常规重放只运行离线检查/适配测试：

```bash
bash /home/zyf/veta_echo_mvp/artifacts/hab10_readiness/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/hab10_readiness/reproduce.sh adapter-tests
```

原SDK及History/Prompter只向自有临时fake receiver发送实拍/真实反馈，假响应不执行，
不能称为模型消费或自主恢复。后续40组独立输入均cap0/未执行，60次仍需新明确授权。
`HAB10_EXECUTION_READY`只表示数据/零调用接口验收，不代表产生HAB科研结果。

V11最终五项READY均PASS，20/20唯一情境验收、25项适配/计数器测试通过。
实际共28次Minecraft case尝试：S2一次、旧工程6成功/1挖掘前失败、统一v2验收20次；不混算独立样本。
63次自有fake HTTP、真实LLM调用0、Agent试次0；既有模型保留，全部自有Java/Bot/Xvfb已清理。
V9/V10只读检查仍通过；最终报告为`artifacts/hab10_readiness/final_decision.md`。

## V12 — HAB-10 P0 Agent Pilot: Stage 0 记录（授权前）

初始任务尚未批准V12的60次新请求，故当时仅完成Stage0，状态
`WAITING_FOR_V12_AUTHORIZATION`。交付目录`/home/zyf/veta_echo_mvp/artifacts/hab10_pilot/`；
263项数据/配置/来源/模型/资源预检和16项隔离预算检查通过，真实推理0、Agent0/40、未生成初始计划。
V9/V10/V11完整树及九核心/源provenance hash不变；仅对coder390做只读Docker/config和两次GET，
没有启动Minecraft、GPU任务、模型网关或假接收端。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_pilot/budget_tests.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_pilot/stage0.py
bash /home/zyf/veta_echo_mvp/artifacts/hab10_pilot/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/hab10_pilot/reproduce.sh budget-tests
```

`stage0.py`一次性，拒绝覆盖现有目录；通常只运行后两条离线重放命令。
新`06_budget_ledger.json`为`CLOSED_UNAUTHORIZED`、authorization=null、active cap0、attempts=[]；
真实拒绝探测不改账本，120个并发正向探测只发生在临时fixture，不是批准或推理请求。
预算原语用flock+原子落盘，失败/重试不退款，20初始/40在线及各unit一次，按实验/phase/run隔离，
所有接口共用总限额；尚未接入实际V12 SDK网关或完整执行器，授权后再实施并冻结。

预算仅提案：V12-HAB10、本地coder390、最多60新请求、输入8192/输出1600、SDK重试0、
API现金上限0元、无外部收费API/新增GPU租用，现有电费折旧排除API预算；旧累计12请求账户已关闭。
必须收到这一范围的新明确批准后才记录授权时间/来源、绑定协议/模型hash并进入Stage1。
本次不生成未执行的采纳/收益/Delta/成功率，40个planned映射的actual_log_path均为null。

### V12 明确授权后的执行

用户后续明确批准本轮本地coder390最多60个新请求：20初始、40在线、每trial最多1次恢复，
输入8192/输出1600、失败计数、SDK重试0、API现金0元、无外部收费API或新增GPU租用。
上述Stage0结果完整保存在`hab10_pilot/stage0_frozen/`，不覆盖历史未授权证据。
实际授权原话、记录时间与模型/协议绑定在`authorization.json`；未用额度随本批关闭。

一次性执行命令如下，已启动后不可重复执行或擅自补跑：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_pilot/authorize.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_pilot/build_runtime.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_pilot/runner.py
```

`pre_model_method_freeze.json`在首个模型请求前固定执行代码，
10次真实无动作初始capture各供同模板N/M共享；20份输入先全部冻结，20份计划各生成一次，
全部封存后才按V10顺序执行40次正式试验。前8次是正式试次的完整性门，不是额外canary。
无效计划、未采纳和正常任务失败保留，不重采样；工程失败则停批且不自动续跑。
独立本地网关18096只向127.0.0.1:8000转发；每请求先持久计费、核验真实RGB和完整多模态token，
原始Echo/Mindcraft核心不修改，只在隔离执行器做mask、生命周期和预算薄适配。

批次退出并清理自有进程后，运行纯离线汇总/核验（不调用模型、不重跑Minecraft）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab10_pilot/finalize.py
bash /home/zyf/veta_echo_mvp/artifacts/hab10_pilot/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/hab10_pilot/reproduce.sh budget-tests
bash /home/zyf/veta_echo_mvp/artifacts/hab10_pilot/reproduce.sh analysis-tests
```

实际完成数、失败、采纳、成本和研究结论只以封存后的`final_decision.md`、
`03_raw_metrics.csv`、`04_per_template_results.csv`及原始模型/环境日志为准。
这些是Echo-Reconstructed的P0先导证据，不是Official Echo或已验证的VETA方法。

V12本批已实际完成40/40正式运行、37/40目标成功，无工程中断；
H07的同一无效初始计划在C-M/T-M各失败一次，H10-T-M一次在线恢复后仍不可达并按恢复上限失败，
均保留且不补跑。7/10模板有跨对象行为采纳；8个四条件均成功的可评模板中，
主成本dig的Control记忆收益0个，Treatment额外成本6个，额外dig合计6、均值0.75。
新本地请求27次（20初始+7在线），实际输入55959/输出6260 tokens，API现金0元；
未用33次已关闭，自有Java/Bot/Xvfb/网关已清理，原coder390保留。
描述性结论为`HAB10_MEMORY_VALUE_WEAK`与`HAB10_LOW_COST_FAILURE`：
存在局部经验失效，但尚不支持直接进入完整VETA或把主成本正迁移当作已证事实。

## V13 — Experience Transfer Utility Headroom Gate

源码`scripts/hab_utility_gate/`，独立交付目录
`/home/zyf/veta_echo_mvp/artifacts/hab_utility_gate/`。本轮LLM硬上限0、没有SDK/模型网关，
不启动新Minecraft或Agent，不使用V12关闭余额，不修改V10/V11/V12。

首次离线构建（拒绝覆盖已有产物）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab_utility_gate/audit.py build
```

该命令逐条读取40个真实controller/actions/服务器库存和环境配置，
用现有`prismarine-nbt`离线解码10个原始世界快照，不连接服务器；
重算Control辅助成本及N/M、逐情境clairvoyant、同模板C/T同选静态上界，
初始计划按请求ID并集去重计费，不把oracle称为新Agent/VETA。
另做重复equip的粒度敏感性诊断，不替换V12主指标。

常规重放只用以下只读检查/单元测试：

```bash
bash /home/zyf/veta_echo_mvp/artifacts/hab_utility_gate/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/hab_utility_gate/reproduce.sh tests
```

未来候选按每个source_key的首个pair_id确定为H01/H02，不看收益大小选对象。
新布局为探索性独立设计，保留两资源在原8格LOS内，利用固定地形产生远替代路径，
四邻接名义路径差12格不冒称原生路径实测。未来最多8个Agent试次/12请求仅为提案，
本轮执行授权与模型cap均0；真实安全求路、机制/RGB和公平输入门尚未运行，
不能将DESIGN_READY当作EXECUTION_READY，更不能并入HAB原结果。

## V14 — Cost-Sensitive Transfer Live Preflight

源码`scripts/hab_utility_preflight/`，真实环境证据写入独立
`/home/zyf/veta_echo_mvp/artifacts/hab_utility_preflight/`。
仅确定性Minecraft/Mineflayer验收，不调用LLM、不运行Agent、不修改V12/V13或世界坐标。
同一V13冻结几何先创建两个独立base，每个C/T与近/远case独立克隆相应base；
使用原MC1.19/Java17/Node22/native链、Renderer/mask/V9观察计数器。

本轮一次性入口（已经执行，不能重新运行覆盖）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab_utility_preflight/stage0.py
PYTHONDONTWRITEBYTECODE=1 python3 /home/zyf/veta_echo_mvp/artifacts/hab_utility_preflight/execution_code/run.py
```

保留两次启动确认超时：初始clear区域已经是air，合法`No blocks were filled`被错误等待逻辑忽略。
只修确认逻辑，在同一已保存世界继续原几何，未重新生成世界；
随后两次初始观察在任何导航/dig前失败：新采集层错误在Vec3原地归一化后取长度，
把原8格LOS变成1.9格。恢复原V11的先取长度再normalize，全8条件统一复验，不改mask或坐标。
两个工程修订、原代码/失败/世界hash/完整diff分别保存在`engineering_revision_r1/`、
`engineering_revision_r2/`，没有删失败或按收益挑结果：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab_utility_preflight/bootstrap_continue.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab_utility_preflight/acceptance_r2.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab_utility_preflight/report.py interface
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab_utility_preflight/report.py
```

这些是有标记的一次性工程/验收入口，不是用户可反复补跑的实验命令。
常规仅作只读重放：

```bash
bash /home/zyf/veta_echo_mvp/artifacts/hab_utility_preflight/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/hab_utility_preflight/reproduce.sh counter-tests
```

真实接受版8/8条件通过，另保留2次初始观察工程失败，共10次确定性Bot session，Agent0。
近/远各1次navigate却有原生路径0/9.656854格、累计物理移动0/约9.543238格、
physics ticks和独立服务器gametime均11/46的差异（含共同10tick停稳窗）；名义12不是实测。
16次原SDK只发自有本地假接收端，真实模型请求0、API现金0，历史关闭预算不动。
这只验证环境可测且公平，不能报告Memory或VETA优势；未来最多12模型请求仍须新明确授权。

## V15 — Cost-Sensitive Experience Transfer Utility Pilot

源码`scripts/hab_utility_pilot/`，独立输出
`/home/zyf/veta_echo_mvp/artifacts/hab_utility_pilot/`。
当前仅Stage 0零调用核验：读取V13/V14真实报告与manifest，保护V12/V13/V14整树、
源认证/九核心/几何/规则/renderer和模型配置，重验历史真实RGB及在线反馈SDK回执，
只读运行mask/counter回归、检查进程/端口/瞬时资源。不重复V14活体验收、不访问模型HTTP。

首次构建拒绝覆盖已有输出：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab_utility_pilot/stage0.py build
```

常规只读检查与离线测试：

```bash
bash /home/zyf/veta_echo_mvp/artifacts/hab_utility_pilot/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/hab_utility_pilot/reproduce.sh tests
```

状态`WAITING_FOR_V15_AUTHORIZATION`：12次（4初始+8在线）是提案而非批准，
独立生产账本authorization=null、cap0，旧V12关闭余额不复用。没有SDK/网关/Agent执行入口，
没有新计划/试次/效用指标；计划映射不是运行记录。未来需V15新明确批准，并完成实际
processor输入计数与独立预算硬门接线、正式执行代码冻结，才可开始固定四计划/八试次。

Stage0初轮两项过严判据完整保存在`stage0_initial_check/`：V13/V14字段顺序不同而JSON值相同；
root模型容器进程的cwd不可读，通过只读Docker top和精确cgroup双重归属排除，未杀进程。
一次性核验修订（不更改环境/模型/科学条件、不可重复）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab_utility_pilot/stage0.py refine-audit-once
```

### V15 新授权执行

用户另行明确批准12次本地请求（4初始+8在线），失败计数、8192/1600、SDK重试0、API现金0元。
独立绑定新账本，不复用旧V12关闭预算。以下为一次性命令，不得重跑或检查点续跑：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab_utility_pilot/prepare.py
PYTHONDONTWRITEBYTECODE=1 python3 /home/zyf/veta_echo_mvp/artifacts/hab_utility_pilot/execution_code/pilot_tests.py
PYTHONDONTWRITEBYTECODE=1 python3 /home/zyf/veta_echo_mvp/artifacts/hab_utility_pilot/execution_code/runner.py
PYTHONDONTWRITEBYTECODE=1 python3 /home/zyf/veta_echo_mvp/artifacts/hab_utility_pilot/execution_code/report.py finalize
```

正式执行器仅从V12薄适配新路径/命名空间/12硬门，加被动物理计数和真实console gametime；
原系统提示、策略、源CSD/五维、renderer/mask、工具和布局不改。先实拍两份新当前观察，
再四计划一次性冻结，按V13顺序执行8个干净trial，C/T共享同一计划；工程异常停批。
Stage0完整归档`stage0_frozen/`，新增源码来源和完整diff随产物冻结。常规仍只使用
`reproduce.sh check|tests|budget-tests`，不自动调用LLM或重跑环境。

V15实际完成：8/8 Agent工程有效且任务成功，4初始+3在线=7真实请求；剩余5次随账本关闭。
仅H02呈现真实路线收益与失效代价；H01只省零移动navigate。结论
`UTILITY_TRADEOFF_OBSERVED / GO_TO_VALIDITY_CRITERION_STUDY`仅支持考虑判据研究，不证明VETA。
同一N计划在H02 C/T存在落物/自动拾取变异，不把全部DeltaShift归因为隐藏规则。

独立的事后只读数值审计（不修改冻结方法或评分）与描述性补充：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab_utility_pilot_analysis/verify.py --save-once
PYTHONDONTWRITEBYTECODE=1 python3 scripts/hab_utility_pilot_analysis/write_addendum.py write-once
```

上述写入命令各只执行一次；常规仅重新核验：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 /home/zyf/veta_echo_mvp/artifacts/hab_utility_pilot/analysis_tools/verify.py
PYTHONDONTWRITEBYTECODE=1 python3 /home/zyf/veta_echo_mvp/artifacts/hab_utility_pilot/analysis_tools/verify_completion.py
PYTHONDONTWRITEBYTECODE=1 python3 /home/zyf/veta_echo_mvp/artifacts/hab_utility_pilot/analysis_tools/write_addendum.py check
bash /home/zyf/veta_echo_mvp/artifacts/hab_utility_pilot/reproduce.sh check
```

补充展开十问、真实初始规划/在线延迟和局部失效，保留`report_before_addendum/`原主报告；
不重复调用`report.py finalize`覆盖补充，物理窗口/主指标/原始指标文件保持不变。

## V16 — Transfer Validity Learnability Gate

源码`scripts/validity_information_gate/`，独立结果
`/home/zyf/veta_echo_mvp/artifacts/validity_information_gate/`。
新增LLM硬上限0，不使用V15关闭额度，不训练模型、不运行Agent或扩HAB。
Stage A逐动作核对V12全部40/V15全部8轨迹、34真实模型请求、原源经验/真实机制证据。
已有10条跨对象恢复不能被误说成没有跨对象互动；缺的是同族探针→封存留出有效性判断结构。

独立配方材料族按原生1.19 recipe及HAB已验收目标的确定性规则选H01 granite→H03 polished_granite，
U=H02 diorite。使用原生可见纹理/名称和公开材料知识；历史遮蔽/renderer/协议文件不改。
C/T显式人为耦合A/B掉落规则，U保持不变，不能称自然关联。先A_C/A_T、封存B/U判断，再干净克隆
B_C/B_T/U_C/U_T。规则基线在独立`python -I -S`进程只读白名单public JSON，无世界真值/时间代理/B结果。

一次性执行（拒绝覆盖/重跑）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validity_information_gate/audit.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validity_information_gate/tests.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validity_information_gate/prepare.py
PYTHONDONTWRITEBYTECODE=1 python3 /home/zyf/veta_echo_mvp/artifacts/validity_information_gate/execution_code/run.py
PYTHONDONTWRITEBYTECODE=1 python3 /home/zyf/veta_echo_mvp/artifacts/validity_information_gate/execution_code/report.py finalize
```

常规离线重放，不重启Minecraft：

```bash
bash /home/zyf/veta_echo_mvp/artifacts/validity_information_gate/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/validity_information_gate/reproduce.sh tests
```

主效应严格按B primary效果，不把fallback任务成功当作动作有效；U的abstain不算正确分类。
同数据五类判据比较仅形成协议，不执行训练/MLLM；单族两个B预测不提供神经优势或统计泛化证据。

V16实际结果：6/6确定性case完成、B的2个先封存预测均与独立干净验证一致，
U保持有效且基线不传播A的失效（abstain不计正确分类）；50项证据门及9项离线测试通过。
结论`CONTROLLED_EVIDENCE_TRANSFER_FEASIBLE / GO_TO_VALIDITY_CRITERION_EXPERIMENT`仅限人工耦合假设。
原生配方只支持材料关联，不证明自然掉落关联；简单规则已能解决这两个B情境。
协议冻结前曾遇到原生recipe的key含备选list，修正为仅选择单个明确材料；旧源码和错误保存在
`preparation_diagnostics/`，当时新服务器/试次/模型调用均0，未按运行结果调整对象或规则。

## V17 — Conditional Evidence Transfer Boundary Test

源码`scripts/validity_boundary_test/`，新输出
`/home/zyf/veta_echo_mvp/artifacts/validity_boundary_test/`；LLM硬上限0，无Agent/训练/新经验族。
只挑战V16的显式有效性耦合，不否定其材料配方事实。新C逐字节复用V16 C，E只替换B原有T
loot，A/U/工具/布局/renderer/源经验保持不变；4个case都克隆同一V16干净base，不改旧世界。

一次性入口（不可重跑或按结果改世界）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validity_boundary_test/audit.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validity_boundary_test/tests.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validity_boundary_test/prepare.py
PYTHONDONTWRITEBYTECODE=1 python3 /home/zyf/veta_echo_mvp/artifacts/validity_boundary_test/execution_code/run.py
PYTHONDONTWRITEBYTECODE=1 python3 /home/zyf/veta_echo_mvp/artifacts/validity_boundary_test/execution_code/report.py finalize
```

原V16动作控制器仅薄适配输出/文本去目标ID及post-A原始/抽象记录，完整diff存档。
原规则函数逐字节不改，adapter明确“材料相关≠已知耦合”，保守对照UNKNOWN/VERIFY；
两者在独立B世界启动前用`python -I -S -B`封存，避免写旧代码pycache。
判据允许视图在执行前固定为初始RGB/别名安全文本+材料关系+A语义效果；原始post-A
RGB/完整状态另存诊断，若不同则明确报告，绝不把抽象一致说成原始多模态一致。

只读重放：

```bash
bash /home/zyf/veta_echo_mvp/artifacts/validity_boundary_test/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/validity_boundary_test/reproduce.sh tests
```

弃权不算正确预测/免费成功，直接B验证及E任务fallback成本分开；U仅复核原有真证据与相同规则，
不新增U实验。不可识别判断只限实际相同的允许视图，不凭规则错误宣称神经优势。

V17实际完成4/4确定性case、45项证据门和10项离线测试，新增LLM请求0。
两世界A均破块并获得1圆石，封存的传播规则均预测B有效；独立验证B_C有效、B_E无圆石，
后者经原合法替代路径完成任务。两份允许输入逐字节相同，但post-A原始RGB、位置/朝向和
掉落实体记录不同，因此`SINGLE_PROBE_TRANSFER_NOT_IDENTIFIABLE`仅限预注册静态观察+
A安全语义反馈，不是完整原始多模态不可识别证明。未确证额外可学习信号，不进入f_phi实现。

## V18 — Conditional Evidence Sufficiency Gate

源码`scripts/conditional_evidence_gate/`，输出
`/home/zyf/veta_echo_mvp/artifacts/conditional_evidence_gate/`。
新增LLM硬上限0，仅读取V16/V17全部10条真实轨迹、原始图像、规则及预测封存记录。
先审计局部B掉落干预的语义因果路径；无合法附加信号时按要求停止，不强制新增C/B共享规则。
本入口不含Minecraft/Agent/模型执行器，不修改旧世界、算法核心或论文。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/conditional_evidence_gate/tests.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/conditional_evidence_gate/audit.py analyze
bash /home/zyf/veta_echo_mvp/artifacts/conditional_evidence_gate/reproduce.sh check
bash /home/zyf/veta_echo_mvp/artifacts/conditional_evidence_gate/reproduce.sh tests
```

`analyze`只允许创建一次全新审计目录；后续使用只读`check/tests`。
若因结构性信息缺失停止，不创建03阶段假轨迹/预测/留出结果；04成本表仅引用历史实测并标明来源，
不把C未执行记为免费成功。原始视觉nuisance和未验证侧信道分别标记，避免虚称完整raw严格一致。

V18实际结果：`NO_IDENTIFIABLE_CONDITIONAL_SIGNAL / NO_GROUNDED_CONDITIONAL_PROBE`。
32项审计门、9项测试及只读重放通过；10条历史轨迹、40张原图和6份seal已核查。
现有B干预只修改loot的item名称；扫描原生3636个data JSON未发现其他B loot引用。
材料配方不能建立额外探针到B的有效性因果路径，按停止条件不造C/B耦合、不启动新世界。
03阶段三份结果文件按规定不存在；新增模型/Agent/探针/训练均0，不建议进入f_phi训练。

## 学习价值门：设计 1

`research/learning_value_gate/01_method.md`定义信息、选择余量、强规则、经验独立贡献和净收益五道门。
唯一新候选是可观察空间条件下的采集程序适用性；原隐藏loot不可识别结论保留，不能混写研究范围。
设计固定8个情境规格和未来有界确定性pilot，显式纳入原生pathfinder及同容量去经验对照。
本轮只读核查已有证据、生成设计，未启动新世界/LLM或训练；具体voxel与执行契约仍需先验收冻结。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 research/learning_value_gate/build_design.py build
PYTHONDONTWRITEBYTECODE=1 python3 research/learning_value_gate/build_design.py check
```

`build`拒绝覆盖已有设计存档，日常只运行`check`。设计状态为
`DESIGN_SPECIFIED_HEADROOM_UNTESTED`，不能把可执行的检验思路称为学习价值已经成立。

源程序补充见`research/learning_value_gate/source_options/notes.md`：已有V15 H01 C-M的真实equip/dig
与V16 A_C的真实navigate/dig/pickup可作为同采集族的不同程序候选，未改S1/S2。
通过`python3 -B research/learning_value_gate/source_options/audit.py check`重查原始效果与hash；
不把不同历史控制器的成本差当作程序因果收益，不自动扩充已记录的pilot预算。

## 相同V16请求的复核

附件`31bce4ae-6956-4105-b5c7-4de700f049fd`与冻结V16请求SHA256相同。
复核脚本只将原Stage A审计输出重定向到独立记录，重新读取48条原始轨迹和34份真实请求，
并复核原六次真实Minecraft实验；不覆盖原V16、不新运行世界/模型。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validity_information_gate/recheck_request.py execute
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validity_information_gate/recheck_request.py check
```

独立记录在`research/request_rechecks/v16_01a6da00bf8b/`，六问回答同时说明V16历史GO与
后续V17/V18边界，不能把重复附件当作已新增独立实验。

## Learning Value Gate — G0 and Environment Readiness

源码`scripts/learning_value_readiness/`，输出`research/learning_value_gate/readiness/`。
先实际读取原设计、审计真实源契约，冻结完整8个voxel布局与共同执行器/规则。
Stage3每场一次真实几何/RGB/路径安全检查、零dig；Stage5另2次固定源式初态的真实执行器工程smoke，
验证dig/拾取/独立库存/V9计数，不计入G1/G2策略收益比较。LLM/训练/48或24正式试次均0。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/learning_value_readiness/freeze.py
PYTHONDONTWRITEBYTECODE=1 python3 research/learning_value_gate/readiness/revision_3/execution_code/run.py
PYTHONDONTWRITEBYTECODE=1 python3 research/learning_value_gate/readiness/revision_3/execution_code/report.py finalize
PYTHONDONTWRITEBYTECODE=1 python3 scripts/learning_value_readiness/readiness_rollup.py build
bash research/learning_value_gate/readiness/reproduce.sh
```

冻结和live入口均拒绝覆盖/重跑；若失败保留结果，不按胜负调资源或预算。
source角色仍LIMITED：P_navigate目标站位绑定为手写候选，不能冒称历史自动归纳或经验独立价值成立。

G0实际就绪版本为`readiness/revision_3/`，八个场景全部真实可达且安全往返，名义替代路线原生实测4/8格。
首轮拐角斜线碰撞和第二轮maxDropDown脚部到floor语义错误均保留原始记录；通用碰撞过滤与一格高度
适配对所有方法共享，三版场景hash完全相同。总23次自建服务器会话含全部工程失败与重验，未当独立研究样本。
当前15项工程测试、两次真实执行器smoke及全部readiness证据检查通过。P_direct工程重放有一次未自动入库、
一次成功，未改变程序或加pickup；随机性不作为适用边界。源契约LIMITED，Native Rule仍潜在充分。
新增LLM/训练/48或24正式比较均0，只有`GO_TO_G1_G2_PILOT`工程资格，不是学习收益结论。

## 下一阶段：强规则优先的学习必要性筛查

设计在`research/learning_necessity_gate/01_next_stage.md`，逐场复核已有8场真实binding/库存/路径/RGB，
不重跑Minecraft。先提案H盲复用、N强No Memory、R经验契约规则各8次，共24次；只有候选优势
才完成另外48次，总上限72。该协议独立于旧48+24提案，不能相加，不是本轮live授权。
本轮新增模型/世界/训练0；N/R在线适配仍待实现和冻结，源契约LIMITED没有被升级。

```bash
python3 -B scripts/learning_necessity_gate/test_gates.py
python3 -B scripts/learning_necessity_gate/audit.py build
python3 -B scripts/learning_necessity_gate/audit.py check
```

`build`仅首次存档；后续只运行`check`。判定器的合成单元测试不是Minecraft试次或学习结果。
首批阴性是资源止损而非不存在的数学证明；规则失误、手写程序成功和少量重复阳性都不能直接证明学习必要。

## Learning necessity screen：真实执行

用户授权继续验证后，执行源码位于`scripts/learning_necessity_live/`，当前独立存档为
`research/learning_necessity_live/v2/`。H复用G0冻结程序，N/R共用新原生闭环；仅R接收真实源前提。
公共wrapper记录所有动作并在长调用期间强制400 ticks/60秒预算，V9 dig和Java库存独立核验。
初始观察在同情境逐字节比较，任何工程异常停止；首24次按冻结gate决定是否补齐其余48次。

```bash
python3 -B scripts/learning_necessity_live/freeze.py
python3 -B research/learning_necessity_live/v2/execution_code/run.py
python3 -B research/learning_necessity_live/v2/execution_code/report.py finalize
python3 -B research/learning_necessity_live/v2/execution_code/report.py check
```

freeze/live/finalize均为一次性入口，完成后只使用check；失败试次不补跑、不覆盖。
本地端口25693，单个自建Java服务，原G0 Bot身份与场景保持；不调用模型、不训练。
v1在Mineflayer插件加载前访问setControlState而初始化失败，未执行动作，已独立封存为ENGINEERING_BLOCKED。
v2在spawn后绑定控制接口，保留原失败及完整新批次顺序，不混算两版结果；全部会话数另计该1次失败。

实际完成v2全部72次：H盲复用22/24成功、65总动作；N强No Memory与R契约规则均24/24成功、59总动作。
最终`NO_ROBUST_SELECTION_HEADROOM_IN_PILOT`，当前候选不进入f_phi训练。R/N的24组首次资源选择相同，
现有源契约未增加决策约束；不把自动入库/拾取波动归因于经验。N/R各有一次额外dig，不宣称所有成本已达下限。
v2运行证据19项检查通过；跨版本共73次服务器会话含v1初始化失败，全部自建进程清理，模型/训练0。
研究决定见`research/learning_necessity_live/06_final_research_decision.md`；逐情境成本与10项轨迹审计在`interpretation/`。

```bash
python3 -B scripts/learning_necessity_interpret.py build
python3 -B scripts/learning_necessity_interpret.py check
```

解释存档已经生成，日常只使用check，不覆盖或重新运行已冻结试次。

## 自主研究机会筛查：反馈归因与规划开销

`research/research_opportunity_gate/`只读重算72次真实策略轨迹与V16/V17全部10次探针，
区分dig、掉落、收集和任务终点，并重放无记忆规划剪枝。属于回顾性探索，不是新增独立实验。
新增模型、训练、世界均0；不修改旧V9计数口径、Echo核心或源经验。

```bash
python3 -B scripts/research_opportunity_gate/test_analyze.py
python3 -B scripts/research_opportunity_gate/analyze.py build
python3 -B scripts/research_opportunity_gate/analyze.py check
```

build只允许首次封存，之后使用check重算并验证输入/代码/输出hash。
学习门、强规则停止条件与未来最多12次零LLM恢复诊断方案见`04_learning_gate_and_next_protocol.md`；
该未来矩阵本轮未运行，不能计作修复成功或经验学习收益。实际结果见`06_research_decision.md`。

## CVPR候选继续筛查：视觉证据链

`research/cvpr_opportunity_v2/`检查V12/V15/V16/V17及HNR72共130条历史结果、389个登记PNG，
并将141条真实掉落记录送入当前原版WorldView/Entities作组件回放。不是新世界/Agent试次。
当前item网格缺失及身份metadata丢失是工程缺口，不是学习收益或CVPR贡献；旧库存真值不受此诊断推翻。
最近邻查重包含CaM、REMM、MemTransfer和主动视觉取证；候选仍需传感验收和独立方法验证。

```bash
/usr/bin/python3 -B scripts/cvpr_opportunity_v2/audit.py build
/usr/bin/python3 -B scripts/cvpr_opportunity_v2/audit.py check
```

使用已有系统Pillow，不改conda或全局依赖。封存后只运行check；它重放组件控制并重算所有PNG/hash/时间窗。
新增模型、训练、新世界均0；不修改旧runtime。研究结论见`research/cvpr_opportunity_v2/06_research_decision.md`。

## 视觉证据可行性与创新否证

`research/visual_evidence_feasibility/`保留独立客户端item纹理适配的实际工程测试与失败。
v1启动1个服务、Bot前缺编码依赖失败；v2完成C世界1次真实dig及8张同帧原版/适配PNG，
但headless Three.js dispose失败，按冻结协议停止，E未执行。不是配对通过或Agent收益。
新工作副本的window动画上下文修复另经实际WebGL组件测试，未重写v2结果或再启动世界。
累计2个新Java会话、1次dig、0模型/训练；全部自建进程清理。旧依赖/世界/算法不修改。

```bash
/usr/bin/python3 -B scripts/visual_evidence_feasibility/review.py build
/usr/bin/python3 -B scripts/visual_evidence_feasibility/review.py check
```

build首次封存后只使用check；不要重跑已关闭的prepare/live。
源码工作副本含清理修复，v1/v2各自execution_code保持冻结；记录彼此差异，不混成一个成功版本。
全文核对发现REMM已有未决状态和缺失证据门，当前候选创新未成立；详见`05_final_research_decision.md`。

## Observation contract audit：研究机会继续筛选

`research/observation_contract_audit/`保存58条原始试次、67个dig/复合窗口的
解码RGB＋动作＋库存/状态前缀审计，输出契约限定的有限记录错误下界。
它不是原Agent性能上限，不改V12/V15主指标，不是新模型实验。
外部AsgardBench/EmbodiedBench仅冻结源码接口核验，未运行外部环境。
候选转向观测契约与评测解释，当前仍非CVPR-ready；模型/训练/Agent/服务器新增均0。

```bash
/usr/bin/python3 -B -m unittest discover -s scripts/observation_contract_audit -p 'test_*.py'
/usr/bin/python3 -B scripts/observation_contract_audit/audit.py check
```

首次构建过程为`audit.py build`，报告完成后`audit.py seal`；现有封存结果不要重建覆盖。
外部只读源码抓取使用`fetch_sources.py`，首次遇EmbodiedBench根LICENSE的HTTP404，
随后使用`fetch_sources.py --resume-partial`保留已下载文件并记录缺失；已封存快照禁止覆盖。
本脚本不执行下载源码、不安装环境、不连接模型；本机系统Python带Pillow即可重放。

## Observation contract live v1：实际启动尝试，尚未验证假设

`research/observation_contract_live/v1/`保留两个环境的工程阻塞：Minecraft在Bot前因
相对入口路径错误退出（Java1次、dig0）；AI2-THOR5.0.0在llvmpipe Vulkan选设备时
崩溃（初始化1次、有效case0）。没有新RGB、Agent试次、模型请求或训练。
Minecraft工作副本路径已修复并加回归测试，冻结失败副本未改；禁止重跑已关闭批次。

隔离环境建立与依赖修正命令（历史执行记录，现有环境不重复安装）：

```bash
/usr/bin/python3 -m venv --system-site-packages --without-pip .runtime/observation_contract/python
.runtime/observation_contract/python/bin/python -m pip install --ignore-installed --no-cache-dir ai2thor==5.0.0 numpy==1.24.4 Pillow==10.4.0
.runtime/observation_contract/python/bin/python -m pip install --ignore-installed --no-cache-dir pip==24.0 urllib3==1.26.20
```

安装后将该独立环境pyvenv.cfg的include-system-site-packages改为false；pip check通过。
冻结完整版本见`setup/requirements.lock`；环境和约798MiB官方引擎归档不提交Git。
download/live实际启动命令为`launch_thor.py download`、`launch_thor.py live`；
当前one-shot ledger禁止重复；HOME/TMPDIR及缓存限定仓库，模型上限0，CPU-only Vulkan。

只读复核：

```bash
/usr/bin/python3 -B -m unittest discover -s scripts/observation_contract_live -p 'test_*.py'
/usr/bin/python3 -B scripts/observation_contract_live/report.py check
```

首次封存使用`report.py build`，封存后只用check。结论见`02_research_decision.md`。

## Observation contract v2/v3：工程恢复与跨引擎真实否证

`research/observation_contract_live/v2/`保留CPU初始化成功、Minecraft Worker路径失败和THOR默认视角空候选停止。
`v3/`通过局部Three构造器context修复、真实Viewer集成测试与独立CPU Vulkan设备选择，完成Minecraft C/E各一次dig及AI2-THOR四case。
Minecraft成对传感修复通过；THOR所选苹果转身后仍可见，未出现预期歧义。不是Agent效用/学习/CVPR创新通过。
THOR准备位姿规则是目标动作前明确冻结的修订；不同reset图像有干扰，不能解释成纯持物因果效应。
本轮实验LLM/训练均0，无外部付费API，不占用GPU；只清理本轮自建进程。

首次运行记录（已关闭，不要重复或覆盖）：

```bash
/usr/bin/python3 -B scripts/observation_contract_v2/launch.py smoke vulkan_index
/usr/bin/python3 -B scripts/observation_contract_v2/minecraft/run.py prepare
/usr/bin/python3 -B /home/zyf/VETA/research/observation_contract_live/v2/minecraft/execution_code/run.py live
/usr/bin/python3 -B scripts/observation_contract_v2/launch.py live vulkan_index
NODE=/home/zyf/.local/opt/node-v22.23.3-linux-x64/bin/node
$NODE research/observation_contract_live/v3/engineering/xvfb/with_xvfb.cjs $NODE /home/zyf/VETA/scripts/observation_contract_v3/minecraft/renderer_test.mjs /home/zyf/VETA/research/observation_contract_live/v3/engineering/renderer_test.json
/usr/bin/python3 -B scripts/observation_contract_v3/minecraft/run.py prepare
/usr/bin/python3 -B /home/zyf/VETA/research/observation_contract_live/v3/minecraft/execution_code/run.py live
/usr/bin/python3 -B scripts/observation_contract_v3/launch.py live vulkan_index
```

组件测试的本地Xvfb wrapper复制自旧`runtime_env/with_xvfb.cjs`，同目录建立`logs/`和指向旧`runtime_env/tools`的只读tools软链接；其生命周期日志独立保存。
新增配置/源码/结果均在本仓库；旧runtime、规则包与world只读复用。THOR复用V1隔离安装，无额外安装。

日常只读复核，不启动新世界：

```bash
/usr/bin/python3 -B -m unittest discover -s scripts/observation_contract_v3 -p 'test_*.py'
/usr/bin/python3 -B scripts/observation_contract_v3/analyze.py check
```

首次分析封存命令为`analyze.py build`、`analyze.py costs`、`analyze.py seal`，均禁止覆盖已有产物。
最终结论：`research/observation_contract_live/v3/04_research_decision.md`；像素/真值复核79项，另存6项回归测试与真实WebGL/Viewer集成验收。

## Transfer identifiability gate：源内容无效对照与事后收益

`research/transfer_identifiability_gate/v1/`对全部72条真实HNR轨迹作回顾性复核，
执行原策略源码重放50次候选选择，并解码核验72张初始RGB。
N/R事后oracle为53动作，固定N/R均59，留一重复选择为63；同一N策略重复配对也出现6动作间隙。
说明oracle间隙不能直接证明可学习的源贡献，不是新Agent结果或新定理；IJCAI2016已有oracle偏差先例。
旧结果不改，新LLM/训练/世界0。不宣称CVPR创新已成立，候选与停止条件见`08_final_decision.md`。

```bash
/usr/bin/python3 -B -m unittest discover -s scripts/transfer_identifiability_gate -p 'test_*.py'
/usr/bin/python3 -B scripts/transfer_identifiability_gate/analyze.py check
/usr/bin/python3 -B scripts/transfer_identifiability_gate/replicas.py check
/usr/bin/python3 -B scripts/transfer_identifiability_gate/visual_audit.py check
```

系统Python3/Pillow和既有私有Node足够，不安装新依赖。首次生成顺序为`analyze.py build`、
`replicas.py build`、`visual_audit.py build`、`analyze.py seal`；已封存后只能check，禁止覆盖。
重配采用精确计数动态规划覆盖6^8种配对，不运行1,679,616个新试次，也不把其当独立样本。

## Prospective A/A：用新真实运行验证源内容无效对照

`research/transfer_null_live/v1/`是独立冻结的A/A校准：八个既有情境各两次，最多16次新Java会话。
两组均执行原N策略，无源输入；A/B只是评测别名，不进入策略配置。共享之前已验收的渲染生命周期及item适配。
这不是此前提出的18次真记忆试验；有效经验源与新模型预算门仍未通过。不得将A/A结果报告成经验收益或视觉模型能力。
结果及停止状态以`outcome.json`、`02_analysis.json`和`04_final_decision.md`为准。失败不替换，LLM/训练上限0。

首次执行命令（一次性批次，不重复运行或覆盖）：

```bash
/usr/bin/python3 -B scripts/transfer_null_live/prepare.py
/usr/bin/python3 -B /home/zyf/VETA/research/transfer_null_live/v1/execution_code/driver.py
/usr/bin/python3 -B scripts/transfer_null_live/analyze.py build
/usr/bin/python3 -B scripts/transfer_null_live/analyze.py seal
```

日常只读复核，不启动模型/世界：

```bash
bash research/transfer_null_live/v1/reproduce.sh
```

沿用已安装Node、Java、私有Xvfb/Mineflayer及系统Pillow；无新增全局依赖。
新端口25698、Bot VetaNull1、独立复制世界；源码和原世界冻结校验，完成后仅清理本次自建进程。

## Execution risk gate：条件风险是否具有稳定预测空间

`research/execution_risk_gate/v1/`只读分析72条HNR和16条后续A/A原始试次。
固定两个合法几何特征，比较简单计数概率与总体频率，分别留情境、留重复和跨批次评价。
后续数据此前已看过，严格标为回顾性探索；预测的是dig返回时未入库，不是规则失效。
未通过稳定增益门，不训练f_phi、不追加世界或LLM。最近邻核验及后续停止门见该目录最终报告。

```bash
bash research/execution_risk_gate/v1/reproduce.sh
```

只需系统Python标准库；首次生成命令为`/usr/bin/python3 -B scripts/execution_risk_gate/analyze.py build`，
封存用`seal`，此后仅运行`check`，禁止覆盖旧产物。提取器一次错误假设及修正保留于`00_extraction_note.md`。

## External source gate：公开历史是否有独立信息

`research/external_source_gate/v1/`冻结EmbodiedMemoryBench公开索引中的前两条Interaction Failure记录，
只读作者JSON与7个源码/许可文件，不运行外部代码或模型。数据来源、许可、固定revision见该目录清单。
历史含任务指令未给出的锁定信息，但属于合成反馈且简单缓存可提取；不冒充自然规律、视觉模型成绩或本地真试次。
源码审查不是完整运行验收；下一步及停止门见`04_overall_research_decision.md`。

日常离线复核（系统Python标准库，无网络/模型）：

```bash
/usr/bin/python3 -B scripts/external_source_gate_analyze.py check
```

首次下载/分析已完成，不重复覆盖：`execution_risk_gate_external_fetch.py`、`external_source_code_fetch.py`，
以及`external_source_gate_analyze.py build`、`external_source_gate_analyze.py seal`。外部文件保留原始作者内容与hash。

## External interface gate：历史约束、执行器与评分语义

`research/external_interface_gate/v1/`继续固定的两条外部案例，冻结60个上游源码/配置/许可文件。
调用原环境分发及评分器做隔离软件测试；spy在模拟器调用边界主动中止，绝不产生伪环境结果。
发现历史合成锁定没有沿所查在线路径传递，按预注册条件停止真实世界阶段。
同时核对论文指标定义和完整baseline历史，避免把合法指标口径或单字段差异误判为bug/信息不公平。
不修改上游、不发模型请求、不安装依赖。上游子项目Apache-2.0许可已独立保存；旧monorepo许可记录不覆盖。

```bash
bash research/external_interface_gate/v1/reproduce.sh
```

当前`python3`为3.14，使用标准库加载上游纯Python接口；运行审计钩子禁止网络和子进程。
首次步骤为`/usr/bin/python3 -B scripts/external_interface_fetch.py`、
`python3 -B scripts/external_interface_audit.py build`和`seal`；已完成后只能`check`，不重复下载覆盖。
源码/协议/结果在manifest中封存。综合判断见`04_final_decision.md`，不能将软件单元输入当Agent试次。

## External contract live：历史约束的真实执行见证

`research/external_contract_live/v2/`用冻结的两个外部案例运行作者原始Env、动作执行器和评分器。
两例历史“Locked”柜体经原合法Navigate→Open路径均closed→open，同时原ERR均为1。
原运行器Open使用forceAction=True、Navigate为Teleport；明确保留该条件，不声称unforced也成功。
这验证公开路径下持续物理锁定解释不能直接成立，不证明评分bug、记忆收益或CVPR创新性。
v1渲染器检测失败（世界0）及后端修复完整保留，没有改作者规则和旧结果。

隔离安装命令（已执行；日常不重装）：

```bash
python3 -m venv .runtime/external_contract/python
.runtime/external_contract/python/bin/python -m pip install --index-url https://pypi.org/simple ai2thor==5.0.0 'numpy>=1.24' 'Pillow>=9' 'jsonschema>=4' 'tqdm>=4'
```

Python3.14.6，准确包版本见v1 `requirements.lock`，共享既有AI2-THOR5引擎，CPU llvmpipe。
首轮运行是`python3 -B scripts/external_contract_live/launch.py 1`及`2`，已完成，不覆盖重跑。
新LLM请求0；每例2个公开动作、10个含初始化RPC；物理ticks未知。仅清理自有进程组。

日常复核与4项提取器单元测试（不启动模型/世界）：

```bash
bash research/external_contract_live/v2/reproduce.sh
```

分析首次命令为私有Python执行`analyze.py build`、`seal`；后续只`check`。
完整证据、使用限制、未来独立重放说明见v2 `03_findings.md`、`04_research_decision.md`、`05_reproduction.md`。

## Constraint semantics gate：非强制控制与独立评分负对照

`research/constraint_semantics_gate/v1/`继续固定的两个柜体，用新独立世界只将native Open的forceAction改为False。
两例仍真实打开、原ERR仍为1。原作者源码不变；这是明确的诊断干预，不是原样官方成绩。
新旧开门前agent/全部对象metadata相等，但RGB不同，不能冒充完全等观察的Agent试验。
独立SafetyALFRED冻结8个源文件，实际运行原评分函数4格软件测试，明确不计作独立物理复现。
现阶段简单类型规则足以诊断边界，不支持训练f_phi或CVPR-ready主张；详见该目录研究决策。

新依赖仅置于独立target，不改旧环境：

```bash
.runtime/external_contract/python/bin/python -m pip install --index-url https://pypi.org/simple --no-deps --target .runtime/external_contract/constraint_semantics_packages pandas==3.0.6
```

首次执行为`python3 -B scripts/constraint_semantics_gate/launch.py 1`及`2`，已完成，不覆盖旧数据重跑。
资源沿用CPU llvmpipe；新LLM请求0、世界2个，每例2公开动作/10底层RPC，ticks未知。仅清理自有进程。
源文件下载边界、首次失败和一轮收窄范围的修正见`00_fetch_note.md`；不可把目标目录列表当全库审计。

## Selection interaction gate：策略交互是否超过重复波动

`research/selection_interaction_gate/v1/`逐条读取既有72条H/N/R和16条同策略A/A真实数据。
新增双因素分解和三折留重复预测，保留所有情境/重复；N/R全成功主比较不混入H失败的低成本。
交互动作预测没有优于加性模型，旧留重复选择仍63动作而固定N/R各59；不支持学习必要性或CVPR-ready。
这是事后离线诊断，不是88次新试验；新模型请求、世界、f_phi训练均0。

```bash
bash research/selection_interaction_gate/v1/reproduce.sh
```

仅Python标准库；8项测试均为数学/边界fixture，不计环境样本。首次`analyze.py build`和`seal`完成后拒绝覆盖，
日常仅`check`重算。原数据545个文件哈希与源码/输出封存；首次读取空LLM列表的类型错误及修复日志单独保留。
最近邻重叠、研究进入条件与未实现目标见`04_novelty_boundary.md`和`05_final_decision.md`。

## Physical adaptation gate：独立物理适应候选

`research/physical_adaptation_gate/v1/`审计公开Gated Memory Policy的Memimic push_cube，固定官方提交与22个源码/配置文件。
真实执行原速度序列/评分函数81次软件调用；15点序数搜索fixture及5项测试通过，**不是物理试次或学习成绩**。
确认教师读取真实摩擦、模型输入边界仍需核验；仅支持进入有限物理信息验收，不支持训练必要性或CVPR-ready。

```bash
bash research/physical_adaptation_gate/v1/reproduce.sh
```

沿用`.runtime/external_contract/python/bin/python`和NumPy2.5.4，无新安装。首次源码获取命令为
`python3 -B scripts/physical_adaptation_gate/fetch.py`及有界信息链补查`fetch.py --boundary`；现已封存，拒绝覆盖。
首次分析使用私有Python运行`audit.py build`、`seal`；日常仅`check`。新模型请求、环境运行和训练均0。

## Physical adaptation readiness：九格真实机器人物理验收

`research/physical_adaptation_readiness/v2/`完成原Memimic push_cube的3摩擦×3速度单推送矩阵。
初始RGB及本体状态一致，同速度命令前缀一致，真实效果不同；成功矩阵呈对角结构。
这是物理信息门通过，不是原六推送成绩、记忆Agent收益或CVPR创新证明。V1的GLFW构造失败完整保留，V2只修复我们自己的探测器生命周期。

隔离安装首次命令（不改变全局环境；已完成，无需重装）：

```bash
CONDA_PKGS_DIRS="$PWD/.runtime/physical_adaptation/conda_pkgs" /home/zyf/miniconda/bin/conda create --prefix "$PWD/.runtime/physical_adaptation/python" --override-channels -c conda-forge --no-default-packages python=3.10.15 pip=25.2 -y
.runtime/physical_adaptation/python/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0
.runtime/physical_adaptation/python/bin/python -m pip install -r scripts/physical_adaptation_readiness/requirements.txt
```

完整实装版本在v1 `requirements.lock`；固定官方源码与资产424文件、Git blob核验在`upstream_manifest.json`。
MuJoCo3.3.5、dm-control1.0.31、CPU llvmpipe；使用既有私有Xvfb二进制，但启动/日志/清理均属本轮新进程。
源获取：`python3 -B scripts/physical_adaptation_readiness/fetch.py`（拒绝覆盖）。
首次执行为`python3 -B scripts/physical_adaptation_readiness_v2/launch.py freeze`及`run cell_01`至`run cell_09`，现均已完成，不覆盖重跑。

日常只读复核，不启动世界或模型：

```bash
bash research/physical_adaptation_readiness/v2/reproduce.sh
```

9次真实单推送、601个控制动作；另有1次失败构造。LLM/策略推理/训练均0。后续新闭环需单独冻结，不将当前开发矩阵当学习测试集。

日常离线重放（含6项新增单元测试，不启动模型/世界）：

```bash
bash research/constraint_semantics_gate/v1/reproduce.sh
```

## Physical feedback gate：原六推送闭环的简单规则基线

`research/physical_feedback_gate/v1/`冻结3个既有开发摩擦×固定中速/开放循环/一步RGB反馈规则，
最多9个真实原Memimic六推送episode；零LLM、零策略网络推理、零训练，CPU llvmpipe。
上游源码不变，新增薄适配统一覆盖真值教师速度并禁用原agent的真值位置速度截断。
高层只读RGB特征和自己的动作，底层仍是原状态控制器，不能称端到端RGB-only；
采用原末三次成功评分，不把单推送算整episode。全部条件是开发集，不是独立泛化验证。

首次执行命令（已有结果时拒绝覆盖，不用于日常重放）：

```bash
python3 -B scripts/physical_feedback_gate/launch.py freeze
for case in episode_01 episode_02 episode_03 episode_04 episode_05 episode_06 episode_07 episode_08 episode_09; do
  python3 -B scripts/physical_feedback_gate/launch.py run "$case" || break
done
.runtime/physical_adaptation/python/bin/python -B scripts/physical_feedback_report.py build
.runtime/physical_adaptation/python/bin/python -B scripts/physical_feedback_report.py seal
```

日常只读离线重算RGB特征、每个封存决策、原judge几何真值、成本和来源哈希，不新建世界：

```bash
bash research/physical_feedback_gate/v1/reproduce.sh
```

原始RGB、逐控制步真值轨迹、独立允许输入决策账本和进程清理记录均保留；
真实结果与研究边界见该目录`03_results.md`、`04_research_decision.md`。

## 4090 的云端任务接口

采用 `云端 HTTPS → Cloudflare 临时隧道 → 本机 jobd（127.0.0.1:8100）`。
直接隧道的 URL **不加 `/jobd` 前缀**。
服务源码为 [`jobd.py`](jobd.py)，客户端为 [`bench/jobd_client.py`](bench/jobd_client.py)，均只依赖 Python 标准库。
部署核验结果见 [`research/remote_4090/configuration_check.json`](research/remote_4090/configuration_check.json)。

本机已有 `/home/zyf/cepo/jobd.py`、tmux 会话 `cepo-jobd` 和权限为 `600` 的
`/home/zyf/.config/cepo/jobd.env`，应复用它们，不重新生成 token、不重复启动监听器。
Token 不写入本仓库、终端命令参数或聊天。停止推理服务须经用户明确确认；当前状态见下文。

2026-10-19 用户已明确要求停止 Qwen3.5-27B，已停止其 Docker 容器 `coder390`
（SGLang，模型配置 `qwen3_5`）。两张 4090 的计算进程均已退出，8000 端口已关闭；
jobd 和 Cloudflare 隧道继续运行。对应的停止及后续恢复命令为：

```bash
docker stop --timeout 30 coder390
# 仅在需要恢复推理、且训练已结束后执行：
docker start coder390
```

仅在一台尚未部署的新 4090 主机上执行以下初始化；已有部署不执行本段：

```bash
umask 077
mkdir -p "$HOME/cepo/workspace" "$HOME/.config/cepo" "$HOME/.local/state/cepo/jobs"
cp jobd.py "$HOME/cepo/jobd.py"
if [ ! -s "$HOME/.config/cepo/jobd.env" ]; then
  python3 -c 'import secrets; print("JOBD_TOKEN=" + secrets.token_urlsafe(32))' > "$HOME/.config/cepo/jobd.env"
fi
chmod 600 "$HOME/.config/cepo/jobd.env"
tmux new-session -d -s cepo-jobd 'bash -lc '\''set -a; source "$HOME/.config/cepo/jobd.env"; set +a; export JOBD_ROOT="$HOME/cepo/workspace"; export JOBD_LOG_ROOT="$HOME/.local/state/cepo/jobs"; exec python3 "$HOME/cepo/jobd.py"'\'''
```

4090 本机验证（密钥从本地文件读取，输出不包含密钥）：

```bash
JOBD_URL=http://127.0.0.1:8100 JOBD_TOKEN_FILE="$HOME/.config/cepo/jobd.env" python3 bench/jobd_client.py health
JOBD_URL=http://127.0.0.1:8100 JOBD_TOKEN_FILE="$HOME/.config/cepo/jobd.env" python3 bench/jobd_client.py smoke
```

`smoke` 上传唯一测试文件、下载比对、运行一个读取该文件的 Python CPU 任务，再核对状态和日志。
文件保留在 jobd 工作目录的 `connection_checks/`，不会调用模型或启动训练。

安装 `cloudflared` 后启动隧道：

```bash
bash scripts/remote_4090/start_tunnel.sh
```

脚本使用独立 tmux 会话 `veta-jobd-tunnel`，只有已连接 Cloudflare 边缘、公共无认证请求为 `401`、
带认证 `/health` 返回 `ok` 后才输出 `JOBD_URL=https://…trycloudflare.com`。
日志和成功 URL 保存在被 Git 忽略的 `.runtime/remote_4090/`；失败的创建地址不算可用 URL。
网络需要能连通 Cloudflare 边缘的 TCP 或 UDP `7844`，仅能访问 HTTPS 注册 API 不够。
参见 [Cloudflare 网络要求](https://developers.cloudflare.com/tunnel/configuration/)。

2026-10-19 已定位并修复：实际 IPv4/IPv6 防火墙均允许出站，但本机透明代理的
`nat OUTPUT → TP_OUT → TP_RULE` 会把 TCP 重定向到 52345，导致 Cloudflare 握手失败。
绕开透明代理后，IPv4 测试超时，IPv6 HTTP/2 TCP 7844 已成功连接 Cloudflare。
启动脚本现在自动完成以下步骤：

- 为运行用户（本机 UID 1002）到 Cloudflare IPv6 边缘网段 `2606:4700:a0::/64`、
  `2606:4700:a8::/64` 的 TCP 7844 添加两条 NAT 豁免规则；不更改全局代理或入站策略。
- 在 `veta-jobd-quick-api` tmux 会话运行本仓库的 `quick_service_proxy.py`，监听
  `127.0.0.1:8110`。它只通过现有本地 HTTP 代理 `127.0.0.1:17890` 访问 HTTPS 注册 API。
  隧道数据连接使用 IPv6 直接连接 Cloudflare，无需跳板机。
- 以 `--protocol http2 --edge-ip-version 6` 启动 cloudflared，再执行公网认证检查。

规则管理依次使用当前 root、可用的免密码 sudo，或本机已有 Docker 管理接口。
Docker 方式复用本地镜像 `d6af32f45a50`，仅启动执行 ip6tables 的临时容器，
启用 `NET_ADMIN`、`NET_RAW`、`SYS_CHROOT`，不启动镜像的模型服务，不下载镜像。
规则修改前的快照保存在 `.runtime/remote_4090/ip6tables-before-*.rules`。
启动时会重新应用规则，因此主机或透明代理重启后重新运行启动命令即可。
单独查看或撤销本项目添加的两条规则：

```bash
bash scripts/remote_4090/cloudflare_egress.sh status
# 仅在停止隧道后撤销；start_tunnel.sh --stop 已自动执行此操作。
bash scripts/remote_4090/cloudflare_egress.sh remove
```

此配置针对当前主机的 IPv6 网络和透明代理；其他主机应先核对当地网络配置。
不需要对外开放本机 8100 入站端口。

云端项目的 Network secrets 按用户给定的界面填写：Name 为 `4090 jobd`，
Allowed websites 为 `*.trycloudflare.com`，Custom header 名为 `Authorization`，Prefix 为 `Bearer`，
Value 从本机受限文件 `~/.config/cepo/jobd.env` 的 `JOBD_TOKEN` 值复制，不能粘贴到聊天。
本机终端不能替代云端项目设置界面的保存操作。客户端允许不设置 `JOBD_TOKEN`，此时不生成认证头，
由云端网络代理注入。保存后在取得新设置的云端会话中执行：

```bash
export JOBD_URL=https://实际验证通过的地址.trycloudflare.com
python3 bench/jobd_client.py health
python3 bench/jobd_client.py smoke
```

此方案通过 jobd 在 4090 本机调用 `http://127.0.0.1:8000/v1`，不把 Cloudflare 的 jobd URL
当作 `LLM_BASE_URL`，也不另开模型隧道。临时隧道重启会换域名，需重新取得通过验证的 URL。
停止本仓库创建的隧道而保留现有 jobd、模型服务：

```bash
bash scripts/remote_4090/start_tunnel.sh --stop
```

旧临时地址失效时，在 4090 上确认 jobd 后重新创建隧道；从输出复制新的 `JOBD_URL`：

```bash
JOBD_URL=http://127.0.0.1:8100 JOBD_TOKEN_FILE="$HOME/.config/cepo/jobd.env" python3 bench/jobd_client.py health
bash scripts/remote_4090/start_tunnel.sh --stop
bash scripts/remote_4090/start_tunnel.sh
```

2026-10-22 已重建并通过公网健康检查。排查时发现 IPv6 默认路由缺失，发送 IPv6
Router Solicitation 后 NetworkManager 收到路由器通告并恢复默认路由；排查添加的两条
临时边缘网段路由已删除。核验记录见
[`tunnel_restart_20261022.json`](research/remote_4090/tunnel_restart_20261022.json)。

如改用跳板机，仓库中的 [`Caddyfile`](Caddyfile) 预留 `/jobd/* → 8100`（剥离前缀）
和 `/v1/* → 8000`（保留前缀）两条认证路由。先在跳板机安全配置同一 `JOBD_TOKEN` 和 `CEPO_DOMAIN`，
再执行 `caddy validate --config Caddyfile --adapter caddyfile` 和 `caddy run --config Caddyfile`。
4090 向用户指定的跳板机建立回环 SSH 转发：

```bash
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:8100:127.0.0.1:8100 -R 127.0.0.1:8000:127.0.0.1:8000 "$JUMP_SSH"
```

跳板机模式才使用 `JOBD_URL=https://域名/jobd` 和 `LLM_BASE_URL=https://域名/v1`。
