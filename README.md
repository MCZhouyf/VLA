# MineMem 阶段 0 备份

本仓库保存 MineMem 阶段 0 的复现代码和实验记录。数据源为 BAAI 项目目录 `/share/project/zhoubowen/zhy/zyf`。

| 路径 | 内容 |
| --- | --- |
| [code/minemem](code/minemem) 和 [env.sh](env.sh) | 环境预检、权重下载校验、GPU 监控、随机试跑、并发吞吐和 MCU 闭环的脚本与配置 |
| [results](results) | 预检与 GPU 采样、三轮随机试跑、六档 CPU 并发、两次单卡闭环、checkpoint 验证、逐步数据、失败与成功日志、[阶段 0 报告](results/REPORT_phase0.md) |
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
