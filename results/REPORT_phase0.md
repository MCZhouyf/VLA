# MineMem 阶段 0 完成报告

**验收日期** 2026-11-21（北京时间）  
**运行位置** BAAI 作业容器，`/share/project/zhoubowen/zhy/zyf`  
**阶段结论** MineStudio 100 步随机试跑、CPU 并发吞吐实测、单卡 JARVIS-VLA-7B MCU 挖石闭环三项均已交付。阶段 1 尚未开始。

| 验收项 | 实测结果 | 主要证据 |
| --- | --- | --- |
| 环境与权重 | Python 3.10、JDK 8、Xvfb、Minecraft 1.16.5 已验证。两个固定提交源码已部署，三个发布权重已校验 | [源码清单](phase0_upstream_sources.csv) · [权重清单](phase0_model_downloads.csv) |
| 100 步随机动作 | 真实模拟器跑满 100 步，**38.50 FPS**。四条 reset 命令和动作往返检查均通过 | [汇总](phase0-random-rDiSZG89/summary.csv) · 视频（BAAI 原件 `results/phase0-random-rDiSZG89/rollout.mp4`） |
| CPU 并发吞吐 | **32 个实例**各完成 100 步，聚合 **459.87 FPS**。64 实例档在图形初始化阶段失败 | [汇总](phase0-capacity-KDJd1VgQ/throughput.csv) · [32 实例逐项记录](phase0-capacity-KDJd1VgQ/workers-032/instances.csv) |
| 单卡 MCU 闭环 | GPU 7 的受控石墙场景第 **16 步**挖到石头，**1.116 FPS**。GPU 6 的自然森林场景运行 500 步，**1.202 FPS**，累计奖励为 0 | 受控场景视频（BAAI 原件 `results/phase0-mcu-gIMHeiv5/rollout.mp4`） · 自然场景视频（BAAI 原件 `results/phase0-mcu-JjgkiRLZ/rollout.mp4`） |

## 1. 阶段 0 做了什么

- 在 BAAI 建立隔离的 Conda 环境，固定源代码、引擎与权重版本，验证 Python 导入、Xvfb 启动和实际游戏 reset。
- 用世界观测验证 `/tp`、`/setblock`、`/fill`、`/give`，记录位置姿态字段，并执行环境动作与 agent 动作的往返检查。
- 跑完 100 步随机策略并保存配置、视频、逐步 CSV 和 reset 证据。保留两个早期随机试次的原始失败记录。
- 分档测试 1、4、8、16、32、64 个实例的 CPU 吞吐，记录每实例启动耗时和同步运行阶段的 FPS。
- 在单张 A100 上把 JARVIS-VLA-7B 的生成结果经官方动作解码器送回 MineStudio，完成自然场景记录和受控挖石验收。归档两次视频、逐步动作 token 和环境奖励。

## 2. 环境、资源与版本

### 2.1 BAAI 运行环境

| 项目 | 实际配置 |
| --- | --- |
| Conda 前缀 | `$WORK/envs/mcvla` |
| Python | 3.10.22 |
| JDK | Zulu OpenJDK 8，`1.8.0_514` |
| CPU | `nproc` 为 256。cgroup `cpu.max` 为 `18000000 100000`，相当于 180 核配额 |
| 内存配额 | `816043786240` 字节，即 760 GiB |
| GPU | GPU 6、7 均为 NVIDIA A100-SXM4-40GB。每次闭环进程只暴露一张卡 |
| 渲染 | Xvfb，模拟器使用 CPU 渲染，`MINESTUDIO_GPU_RENDER=0` |
| 游戏引擎 | Minecraft 1.16.5，由运行 JAR 内的 `version.json` 确认 |
| 主要运行库 | `minestudio 1.1.5`、`torch 2.5.1`、`transformers 4.49.0`、`accelerate 1.2.1`、`numpy 1.26.4`、`gymnasium 1.4.0` |
| 图像和下载依赖 | `av 13.1.0`、`opencv-python 4.8.0.74`、`opencv-python-headless 4.8.0.74`、`huggingface-hub 0.28.1` |

`env.sh` 把运行文件、缓存和临时目录集中到 `$WORK` 下。随机试跑和吞吐测试均屏蔽 CUDA；模型试验按设备 UUID 暴露目标卡。远端直连清华 PyPI、清华 Conda 和 HF Mirror 元数据接口可用。本机连接 BAAI 使用 HTTP CONNECT 代理，和远端的下载配置独立。[环境配置](../env.sh) · [早期预检 CSV](phase0_preflight.csv)

阶段初期 GPU 6、7 各有一个占卡进程，PID 在当前容器的进程视图中不可见。空闲监控每 60 秒定向检查这两张卡。卡空闲后开始单卡验证，测试期间终止的外部占卡进程数为 0。试验结束复查时，GPU 6、7 均为 **0 MiB、0%**。

### 2.2 源码与引擎

远端 HF Mirror、ModelScope 和 Gitee 的同名源码候选接口均未提供仓库。继续执行时从本机取得两个官方固定提交的归档，再上传并比对归档 SHA-256。MineStudio 使用 editable 模式安装；JARVIS-VLA 以源码方式提供官方动作解码器。运行时使用 MineStudio 固定的 PyAV。

| 组件 | 固定版本 | 位置或校验 |
| --- | --- | --- |
| MineStudio | `278aa8553668d591339dbf30d281594ed06ee882` | `$WORK/code/MineStudio`，归档 SHA-256 `776aceae125b2643a6c46a64d485b58547a24613cd8fc5493df01afbc42b0e0f` |
| JARVIS-VLA | `0aafd2ed0ba9270f40d710440ca70cdc3fa48a11` | `$WORK/code/JarvisVLA`，归档 SHA-256 `94ba65c4ecf2e0417bbb954dc08d6d727021e101081bcbb30a85d0cd1f0f1fb2` |
| CraftJarvis/SimulatorEngine | `48d4809cfddc7e2b85295e8c39b3c5e8c6d46ae7` | `$WORK/runtime/MineStudio/engine.zip`，458,106,630 字节 |
| 引擎 ZIP SHA-256 | `293fac6ac72245b3365dce0e8bfbb6396fb94df29b23b6538f3bd7e2eec13ec6` | 解包得到 `engine/build/libs/mcprec-6.13.jar`，466,027,889 字节 |

[源码归档校验 CSV](phase0_upstream_sources.csv) · [MineStudio 固定源码](https://github.com/CraftJarvis/MineStudio/tree/278aa8553668d591339dbf30d281594ed06ee882) · [JARVIS-VLA 固定源码](https://github.com/CraftJarvis/JarvisVLA/tree/0aafd2ed0ba9270f40d710440ca70cdc3fa48a11) · [引擎发布快照](https://huggingface.co/CraftJarvis/SimulatorEngine/tree/48d4809cfddc7e2b85295e8c39b3c5e8c6d46ae7)

### 2.3 已校验权重

文件数量和字节数涵盖固定快照内的权重及配置。验证脚本检查文件大小，并对由 LFS 发布的大文件逐一比对 SHA-256。

| Checkpoint | 固定修订 | 已校验文件 | 总字节数 | 本阶段用途 |
| --- | --- | ---: | ---: | --- |
| JARVIS-VLA-Qwen2-VL-7B | `27e83fb89ddf9fe2a3340b9db0a85ba9c3b9c971` | 17 | 16,598,895,925 | 单帧检查与单卡 MCU 闭环 |
| Qwen2-VL-2B-Instruct | `1860c46f12b7449d0f5b426a7474ab715342bc65` | 14 | 4,429,622,940 | 已准备供后续阶段使用 |
| MineStudio STEVE-1 | `34e18c2fd00058deb371b6c34b183eb52de8166c` | 4 | 1,641,921,374 | 已准备供阶段 1 baseline 使用 |
| ROCKET-2 | `1x-22w`、`1.5x-17w` 候选 | — | — | 三个已检查镜像的同名候选接口返回 404，权重仍待取得 |

[权重版本及字节数原始记录](phase0_model_downloads.csv) · [下载与校验脚本](../code/minemem/download_phase0_models.py) · [ROCKET-2 发布仓库](https://github.com/CraftJarvis/ROCKET-2)

## 3. MineStudio 随机试跑

### 3.1 三轮运行结果

三轮均使用种子 `19961103`，都执行了真实模拟器的 100 步随机动作。第一、二轮用于调试命令效果的验证，第三轮满足验收。

| 试次 | 步数 | 实测 FPS | `/tp` | `/give` | `/setblock` | `/fill` | 动作往返 | 结论 |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `phase0-random-k8Vo42xr` | 100 | 32.75 | 通过 | 通过 | 未验证 | 未验证 | 通过 | 相对坐标的条件查询缺少库存标记 |
| `phase0-random-VLhRqiy7` | 100 | 37.73 | 通过 | 未验证 | 未验证 | 未验证 | 通过 | 单步探测的体素列表为空，该次观测没有库存标记 |
| `phase0-random-rDiSZG89` | 100 | **38.50** | 通过 | 通过 | 通过 | 通过 | 通过 | 验收通过 |

第三轮的动作采样加环境步进共耗时 **2.5973 秒**。视频为 **224×224、20 FPS、101 帧**，其中第一帧为 reset 检查完成后的起始画面。正式视频已逐帧解码检查。统计 FPS 时排除了命令探测、世界启动和视频写入耗时。前两轮的原始目录和汇总保留在 BAAI。[通过试次 YAML](phase0-random-rDiSZG89/config.yaml) · [汇总 CSV](phase0-random-rDiSZG89/summary.csv) · [逐步 CSV](phase0-random-rDiSZG89/steps.csv) · 视频（BAAI 原件 `results/phase0-random-rDiSZG89/rollout.mp4`）

### 3.2 Reset 命令与观测证据

`CommandsCallback.after_reset` 实际调用 `sim.env.execute_cmd` 逐项发送命令。第三轮从 reset 的 `player_pos` 得到绝对方块坐标，在 Minecraft 1.16.5 中用 `/execute if block` 作条件查询。配置允许最多 12 个空动作探测步，以便服务端更新状态。第 3 个探测步取得全部标记。[回调源码](https://github.com/CraftJarvis/MineStudio/blob/278aa8553668d591339dbf30d281594ed06ee882/minestudio/simulator/callbacks/commands.py) · [底层命令入口](https://github.com/CraftJarvis/MineStudio/blob/278aa8553668d591339dbf30d281594ed06ee882/minestudio/simulator/minerl/env/_singleagent.py)

| 操作 | 验证办法 | 所见结果 |
| --- | --- | --- |
| `/tp @p ~ ~ ~ 45 0` | 读取 `info["player_pos"]` | `yaw=45.0`、`pitch=0.0` |
| `/give @p minecraft:emerald 1` | 读取库存 | 第 1 个探测步出现 `emerald` |
| `/setblock -6238 68 -4518 minecraft:gold_block` | 指定坐标确为金块时才发放钻石作为标记 | 第 2 个探测步出现 `diamond` |
| `/fill -6237 68 -4518 -6236 68 -4518 minecraft:iron_block` | 两端均为铁块时才发放青金石块作为标记 | 第 3 个探测步出现 `lapis_block` |

通过试次的 reset 位置为 `(-6237.5, 64.0, -4517.5)`。实测 `player_pos` 含 `x、y、z、yaw、pitch`。三个探测步的体素列表均为空。方块命令的验收依据来自两个条件查询触发的库存变化，体素接口本身仍待验证。[实际命令、姿态和库存原始证据](phase0-random-rDiSZG89/reset_evidence.json)

### 3.3 动作空间

- `action_type='env'` 使用按键字典和连续相机向量 `[pitch,yaw]`。`action_type='agent'` 使用按钮与相机的分层离散索引。
- `env_action_to_agent_action` 的输入带批次维度，`agent_action_to_env_action` 把离散索引转回环境动作。用按住前进键和相机输入 `[2.0,-3.0]` 的代表动作核对了 `env → agent → env → agent`，前后按钮和相机量化索引一致。
- MineStudio 默认相机布局每轴 11 档。JARVIS-VLA 使用每轴 21 档，最大绝对角度 10，步长 1，μ-law 参数 μ=20。这个 MineStudio 版本公开的 agent `action_space` 相机维度固定为 121；本阶段将官方解码器生成的动作直接交给已按 JARVIS 配置初始化的环境。
- 三个通过证据检查的 rollout 逐步记录均包含 `x、y、z、yaw、pitch`，后续可直接读取游戏提供的方向字段。

[MineStudio 环境实现](https://github.com/CraftJarvis/MineStudio/blob/278aa8553668d591339dbf30d281594ed06ee882/minestudio/simulator/entry.py) · [JARVIS-VLA 相机配置](https://github.com/CraftJarvis/JarvisVLA/blob/0aafd2ed0ba9270f40d710440ca70cdc3fa48a11/jarvisvla/evaluate/config/mine/base.yaml)

## 4. CPU 并发吞吐

### 4.1 协议与计时口径

1. 依次运行 1、4、8、16、32、64 实例。每个档位测试一次，每个实例计划执行 100 步，种子为 `19961103 + worker_index`。
2. 每个实例使用独立工作目录和 Xvfb，CUDA 被屏蔽，Java 用 `-XX:ActiveProcessorCount=2` 设定有效 CPU 数。OpenMP、BLAS、OpenCV 和软件渲染线程按脚本限制。实例共享读取同一引擎 JAR。
3. 所有实例完成 reset 后发出统一开始信号。聚合 FPS 等于总完成步数除以最早开始步进到最后结束步进的墙钟时间。冷启动耗时单列。这组运行不编码视频。
4. 启动等待配置为 300 秒。任一工作进程提前退出也会使该档停止。同步步进的超时配置为 180 秒。所有实例各完成 100 步的档位计入实测吞吐。

[实际运行 YAML](phase0-capacity-KDJd1VgQ/config.yaml) · [测试脚本](../code/minemem/phase0_capacity.py)

### 4.2 各档实测值

| 并发实例 | 完成实例 | 总步数 | 同步阶段耗时（秒） | 聚合 FPS | 相对单实例 | 启动耗时中位数 / 最大值（秒） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 100 | 5.121 | 19.53 | 1.00× | 57.97 / 57.97 |
| 4 | 4 | 400 | 5.589 | 71.57 | 3.66× | 63.97 / 65.39 |
| 8 | 8 | 800 | 5.717 | 139.94 | 7.17× | 77.48 / 88.22 |
| 16 | 16 | 1600 | 5.985 | 267.34 | 13.69× | 100.45 / 104.97 |
| 32 | 32 | 3200 | 6.959 | **459.87** | **23.55×** | 141.86 / 146.84 |
| 64 | 0 | 0 | — | — | — | 启动失败 |

**32 是本轮通过的最高并发档位。** 32 档的聚合吞吐折合每实例 14.37 FPS，独立实例 FPS 的中位数为 17.60。前者计入整个同步批次的尾部耗时，后者采用各实例自己的运行窗口。更高并发的稳态上限尚未测得。随机录视频测试的 38.50 FPS 使用默认 JVM 配置，容量测试同协议下的单实例基线为 19.53 FPS。[全部档位 CSV](phase0-capacity-KDJd1VgQ/throughput.csv) · [1 实例](phase0-capacity-KDJd1VgQ/workers-001/instances.csv) · [4 实例](phase0-capacity-KDJd1VgQ/workers-004/instances.csv) · [8 实例](phase0-capacity-KDJd1VgQ/workers-008/instances.csv) · [16 实例](phase0-capacity-KDJd1VgQ/workers-016/instances.csv) · [32 实例](phase0-capacity-KDJd1VgQ/workers-032/instances.csv)

64 档有 64 条 Minecraft 启动记录，启动门槛处没有实例完成 reset。工作进程 `013`、`017`、`058` 的日志记录了 `Failed to initialize GLFW`，错误码 `0x10008`。容器 OOM 和 OOM kill 计数均为 0。现有证据定位到图形初始化阶段，更细的根因未验证。失败后按批次工作目录及 CPU 环境核对进程归属，向 206 个该批次残留进程发出 `SIGTERM`，复查其全部退出。后续的按批次清理逻辑已在实际 Conda 环境验证。原始错误保留在远端 `$WORK/results/phase0-capacity-KDJd1VgQ/workers-064/worker-{013,017,058}/worker.log`。

## 5. 单卡 JARVIS-VLA-7B MCU 闭环

### 5.1 模型与执行链

用已校验的 7B 快照做 BF16 推理，attention 后端为 SDPA。模型每步接收当前 RGB 画面和最多两帧历史，按 `Mine stone underground.` 提示贪心生成，输出上限是 64 token。官方 `OneActionTokenizer` 解码后调用 `sim.step`。逐步日志保存下一帧对应的实际奖励、位置朝向和该步生成的 token ID。

论文正文采用 51 个 token 的表述。公开 checkpoint 与官方解码器的实用映射包含 31 个键鼠分组值、42 个相机分箱值和 2 个动作界符，共 75 个控制与界符 ID。本轮运行使用的动作起止 ID 为 `151835`、`151836`，解码器报告的相机配置与公开映射一致。[论文](https://arxiv.org/html/2503.16365v2) · [官方动作解码器](https://github.com/CraftJarvis/JarvisVLA/blob/0aafd2ed0ba9270f40d710440ca70cdc3fa48a11/jarvisvla/inference/action_mapping.py) · [权重 token 映射](https://huggingface.co/CraftJarvis/JarvisVLA-Qwen2-VL-7B/blob/27e83fb89ddf9fe2a3340b9db0a85ba9c3b9c971/added_tokens.json)

实际闭环前曾在 GPU 6 使用一帧归档画面进行检查。该检查使用既有只读运行环境中的 PyTorch 2.6.0+cu124、Transformers 4.57.3 与快速图像处理器。模型加载 17.847 秒，单次生成 2.112 秒，峰值 PyTorch 显存分配 16,283 MiB，得到一组完整动作界符。下表的正式闭环使用项目环境和实时游戏画面。[单帧 YAML](phase0_checkpoint_smoke.yaml) · [单帧 CSV](phase0_checkpoint_smoke.csv)

### 5.2 两次真实闭环

| 项目 | 自然森林场景 | 受控石墙场景 |
| --- | --- | --- |
| 运行 ID | `phase0-mcu-JjgkiRLZ` | `phase0-mcu-gIMHeiv5` |
| GPU | 6 | 7 |
| 固定种子 | 19961103 | 19961103 |
| 初始设置 | 使用上游挖石任务的森林出生设置和初始石铲、石镐 | 放置近距石墙，预先装备石镐。reset 布景和 6 步稳定后交给模型 |
| 最大控制步数 | 500 | 150 |
| 实际执行 | 500 步 | 16 步 |
| 含完整动作界符的步数 | 500/500 | 16/16 |
| 控制循环累计计时 | 415.9086 秒 | 14.3401 秒 |
| 控制循环实测 FPS | **1.202** | **1.116** |
| 终止条件 | 达到步数上限 | 第 16 步触发 `mine_block:stone` |
| 累计环境奖励 | 0 | 1 |
| MP4 | 640×360，20 FPS，501 帧 | 640×360，20 FPS，17 帧 |
| 原始记录 | [配置](phase0-mcu-JjgkiRLZ/config.yaml) · [逐步 CSV](phase0-mcu-JjgkiRLZ/steps.csv) · [汇总 CSV](phase0-mcu-JjgkiRLZ/summary.csv) · 视频（BAAI 原件 `results/phase0-mcu-JjgkiRLZ/rollout.mp4`） | [配置](phase0-mcu-gIMHeiv5/config.yaml) · [逐步 CSV](phase0-mcu-gIMHeiv5/steps.csv) · [汇总 CSV](phase0-mcu-gIMHeiv5/summary.csv) · 视频（BAAI 原件 `results/phase0-mcu-gIMHeiv5/rollout.mp4`） |

控制循环 FPS 的分母是图像预处理、模型生成、动作解码及环境步进的逐步耗时之和。模型加载、世界启动、reset 稳定和视频写入均在此计时口径之外。视频按 20 FPS 编码，实际控制循环速度从逐步 CSV 复算。

自然试跑的水平坐标共有 7 组取值，首个控制步和末步相距约 0.21 方块。该次形成完整的 500 步控制记录，累计奖励为 0。受控复测在同种子世界中构造近距石墙，初始玩家面向墙体，石镐放在当前快捷栏。模型接管后 16 步均从实时画面生成动作，最后一步由环境报告挖石奖励 1。这个结果支持闭环验收。自然场景表现保留为独立试次，当前没有可据此估计的自然场景成功率。[受控场景实际布景命令](phase0-mcu-gIMHeiv5/reset_evidence.json) · [上游挖石任务](https://github.com/CraftJarvis/JarvisVLA/blob/0aafd2ed0ba9270f40d710440ca70cdc3fa48a11/jarvisvla/evaluate/config/mine/mine_stone.yaml)

受控复测前 GPU 6 的二次空闲检查未通过，串行运行器切到 GPU 7。外层日志保留了首次空闲检查的返回码 `75`。实际成功以受控试跑的 `summary.csv` 和逐步奖励为准。复跑入口现已在启动日志和汇总 CSV 显式记录物理 GPU 编号。

## 6. 复跑命令和产物

在 BAAI 进入项目目录后按下列命令复跑。每次运行创建新的结果目录。两个 MCU 命令均打开二次空闲检查，按实测使用的物理卡号分别运行。

```bash
export WORK=/share/project/zhoubowen/zhy/zyf
export GPU_IDS=6,7 CUDA_VISIBLE_DEVICES=6,7 MINEMEM_PROXY=
cd "$WORK"
source env.sh

bash code/minemem/phase0.sh check
bash code/minemem/phase0.sh random
MINEMEM_REQUIRE_IDLE=1 bash code/minemem/phase0.sh mcu
GPU_IDS=7,6 CUDA_VISIBLE_DEVICES=7,6 MINEMEM_REQUIRE_IDLE=1 \
  bash code/minemem/phase0.sh mcu --scene visible_stone
bash code/minemem/phase0.sh capacity 1 4 8 16 32 64
```

| 产物 | 用途 |
| --- | --- |
| [README](../README.md) · [启动入口](../code/minemem/phase0.sh) · [实验 YAML 模板](../code/minemem/phase0.yaml) | 安装和重跑 |
| 随机视频（BAAI 原件 `results/phase0-random-rDiSZG89/rollout.mp4`） · [reset 证据](phase0-random-rDiSZG89/reset_evidence.json) | 核对 100 步、四条命令和位姿 |
| 自然场景视频（BAAI 原件 `results/phase0-mcu-JjgkiRLZ/rollout.mp4`） · 受控成功视频（BAAI 原件 `results/phase0-mcu-gIMHeiv5/rollout.mp4`） | 复核两次单卡闭环 |
| [容量总表](phase0-capacity-KDJd1VgQ/throughput.csv) · [32 实例逐项记录](phase0-capacity-KDJd1VgQ/workers-032/instances.csv) | 复算吞吐 |
| [权重快照清单](phase0_model_downloads.csv) · [源码归档清单](phase0_upstream_sources.csv) | 核对软件及权重版本 |

BAAI 项目目录保留了 `results/phase0-conda-Qt1VMTyl.log`、`results/phase0-watch-dependencies.log`、`results/phase0-random-feedback-TDN6DYDD.log`、`results/phase0-gpu-watch-slDEg1Wn.log`、`results/phase0-mcu-visible-VeKMOgdO.log` 和 `results/phase0-capacity-2B9eI6ju.log`。三个正式视频已通过帧数和解码核查，逐步 CSV 的行数、奖励与耗时已与汇总结果交叉验证。

## 7. 交接和下一阶段

**阶段 0 的三项验收证据已齐全。** 64 实例的图形初始化失败、自然森林场景的单次失败和体素接口尚未验证均已单列。ROCKET-2 的可加载发布权重与配套配置仍待取得，关系到阶段 1 的完整 baseline。

阶段 1 按原方案预计 **1–2 周**。先固定 T1 的地标、相同的教师出发段、返回判定和留出场景，再按 10、30、60、100 格四档评估。目标条件为水平距离不超过 3 格，返回时限为教师最短回程步数的 2 倍加 200 步。每档计划 30 个种子，评测随机策略、VPT、STEVE-1、ROCKET-2 和 JARVIS-VLA-7B，保存逐回合 CSV 并给出带 95% 置信区间的距离曲线。阶段 0 未调整 T1 协议、种子或测试集，也未开展训练。
