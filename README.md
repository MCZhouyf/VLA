# MineMem 复现与实验备份

本仓库保存 MineMem 阶段 0、阶段 1 和新版 T1 阶段 3 实机运行的复现代码与实验记录。数据源为 BAAI 项目目录 `/share/project/zhoubowen/zhy/zyf`。

| 路径 | 内容 |
| --- | --- |
| [code/minemem](code/minemem) 和 [env.sh](env.sh) | 环境预检、权重下载校验、GPU 监控、随机试跑、并发吞吐和 MCU 闭环的脚本与配置 |
| [results](results) | 预检与 GPU 采样、三轮随机试跑、六档 CPU 并发、两次单卡闭环、checkpoint 验证、逐步数据、失败与成功日志、[阶段 0 报告](results/REPORT_phase0.md) |
| [阶段 1 报告](results/REPORT_phase1.md) | JARVIS-VLA 门槛、T1 教师验收、种子冻结、双卡四实例容量和复跑记录 |
| [新版 T1 与阶段 3 报告](results/REPORT_phase3.md) | 随机终点、捷径复核、CPU 零样本试点、数据索引及三个 LoRA 变体 |
| [T1 死亡与重生复核](results/REPORT_t1_respawn.md) | 重生负例、冻结教师与捷径验收、VPT-BC 轨迹复跑 |
| [阶段 3 实机报告](results/REPORT_phase3_gpu.md) | 三组 BF16 训练与闭环、公开权重试点、5 格导航门槛与正式训练快照 |
| [阶段 3 复核与重采集报告](results/REPORT_phase3_followup.md) | 四档可见性与捷径复核、第二版布景验收、重采集、正式训练快照 |
| [阶段 3 第二版启动与恢复验收](results/REPORT_phase3_screened_launch.md) | 本地归档校验、断点恢复、GPU 训练进度和 48 小时调度 |
| [阶段 3 第二版布景训练与评测报告](results/REPORT_phase3_screened_training.md) | 无记忆与空间记忆的最终训练、全量开环验证、闭环导航门槛和权重清单 |
| [GPU 6 无记忆训练报告](results/REPORT_phase3_no_memory_gpu6.md) | 实机时间线、逐步轨迹、验证波动和失败回合分析 |
| [阶段 3 正式结果与分析](results/REPORT_phase3_closeout.md) | 正式 30 格止损、失败机制、位姿折返和 RGB 检索对照、归档复算 |
| [遮挡建筑 CPU 审计](results/REPORT_phase3_structure_audit.md) | 90 个冻结场景、配对可见性、找结构对照和死亡取证 |
| [第二版布景数据索引](results/data/t1-screened-hzw7nYp7) | 1,124 个回合的非图像元数据、采集日志与冻结索引校验 |
| [新版轨迹元数据](results/data/t1-v4-L6ysa9jV/metadata) | 训练、验证与测试的 1,194 个原始回合的布景、位姿与动作 token |
| [logs](logs) | Minecraft 进程原始日志 |
| [BACKUP_MANIFEST.csv](BACKUP_MANIFEST.csv) | 文件大小及 SHA-256 |

视频和图像原件保留在 BAAI 对应的路径。上游源码、引擎和权重可按 [源码版本清单](results/phase0_upstream_sources.csv)、[权重快照清单](results/phase0_model_downloads.csv) 及下方安装步骤取回。Conda 环境、渲染缓存和模拟器运行数据库由环境或实例生成。

## 环境与复跑

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

以下命令复现阶段 1 的原始验收。2026-11-21 复核后的 T1 试点和主对比使用下一节的新版入口。T1 教师用 CPU 运行，可和这组检查并行。以下命令从 BAAI 项目目录执行，沿用阶段 0 的引擎、Conda 环境和权重。[阶段 1 验收报告](results/REPORT_phase1.md)

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

`--cache-vision` 复用冻结视觉编码器输出，保持历史帧数和图像分辨率。`--validation-batch-size` 一次前向处理多个带独立掩码的样本。正式评测用 `t1_screened_fixtures.py --scores ACCEPTANCE_CSV --seeds-file SEEDS --output FIXTURES` 冻结已验收的布景，再将 `FIXTURES` 传给 `t1_exec.sh eval --mode formal --screened-fixtures`。同时传入 `--no-memory-gate`、`--offset-gate` 和 `--forward-control` 三份门槛 CSV。`--rollout-batch-size` 可在同一次模型生成中推进多个独立环境。

`t1_screened_eval.sh {gates|d30|remaining|all} VARIANT RUN FORWARD_CSV` 使用已冻结的场景与导航控制。先测无记忆的两版门槛。`d30` 使用 30 个固定种子，正式回程评测需要结构对照通过。批量闭环的实际聚合吞吐写入 `rollout_timing.csv`。

[第二版启动与断点恢复验收](results/REPORT_phase3_screened_launch.md)

### GPU 6 和 GPU 7 运行监督

监督程序每 30 秒读取训练、验证、检查点和显存状态，直到两组训练结束。已完成运行的采样与异常记录见[监督日志](results/phase3-speed-xOn8Q5je/gpu-supervisor)。

~~~bash
run="$WORK/results/phase3-speed-xOn8Q5je"
cat "$run/gpu-supervisor/latest.json"
tail -n 10 "$run/gpu-supervisor/events.jsonl"
~~~

[阶段 3 第二版布景双卡训练与评测报告](results/REPORT_phase3_screened_training.md)


## 阶段 3 正式 30 格结果

共享 LoRA 的两个变体在相同的 30 个冻结种子上各为 0/30。两组按预定规则止损。代码和非图像数据包括训练及验证曲线的 CSV、各轮续训和诊断、完整数值轨迹、冻结索引及从原始 ZIP 提取的动作与位姿元数据。完整训练和模拟复跑依赖服务器上原有的视觉归档与权重。 [正式结果与分析](results/REPORT_phase3_closeout.md) · [原始日志与控制修复](results/REPORT_phase3_control_diagnostics.md) · [归档清单](results/phase3_archive_inventory.json)

```bash
python3 -B code/minemem/t1_phase3_closeout.py \
  --run results/phase3-steering-paired-2AGguG2e/shared-adapter-probe/d30 \
  --fixtures results/phase3-speed-xOn8Q5je/screened_fixtures.json
```

## 阶段 3e 分层空间记忆

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
