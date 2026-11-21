# MineMem 阶段 0 与阶段 1 备份

本仓库保存 MineMem 阶段 0 与阶段 1 的复现代码和实验记录。数据源为 BAAI 项目目录 `/share/project/zhoubowen/zhy/zyf`。

| 路径 | 内容 |
| --- | --- |
| [code/minemem](code/minemem) 和 [env.sh](env.sh) | 环境预检、权重下载校验、GPU 监控、随机试跑、并发吞吐和 MCU 闭环的脚本与配置 |
| [results](results) | 预检与 GPU 采样、三轮随机试跑、六档 CPU 并发、两次单卡闭环、checkpoint 验证、逐步数据、失败与成功日志、[阶段 0 报告](results/REPORT_phase0.md) |
| [阶段 1 报告](results/REPORT_phase1.md) | JARVIS-VLA 门槛、T1 教师验收、种子冻结、双卡四实例容量和复跑记录 |
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

阶段 1 先验证 JARVIS-VLA 的公开任务复现和 5 格导航，再进入 T1 评测。T1 教师用 CPU 运行，可和这组检查并行。以下命令从 BAAI 项目目录执行，沿用阶段 0 的引擎、Conda 环境和权重。[阶段 1 验收报告](results/REPORT_phase1.md)

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
