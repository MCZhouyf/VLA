# 阶段 3 闭环控制复核

## 正式 30 格评测结论

北京时间 11 月 25 日 00:36，共享 LoRA 的正式 30 格评测正常结束。无记忆和空间记忆组在相同 30 个冻结种子上均为 **0/30**，各自的 Wilson 95% 置信区间为 0%–11.35%。60 场均耗尽步数预算，44,296 个动作全部完整解码，死亡和位姿跳变均为零。无记忆和空间记忆组的最近距离中位数分别为 28.381 和 29.818 格。空间组需要至少 16/30 才能继续，最终判定为 `proceed:false`。共享 LoRA 的其余距离正式评测和新版长窗口的完整训练与评测均未启动。 [正式判定](phase3-steering-paired-2AGguG2e/shared-adapter-probe/d30/early-decision.json) · [无记忆逐回合结果](phase3-steering-paired-2AGguG2e/shared-adapter-probe/d30/no_memory/episodes.csv) · [空间记忆逐回合结果](phase3-steering-paired-2AGguG2e/shared-adapter-probe/d30/spatial/episodes.csv)

## 闭环控制修复记录

运行目录为 `results/phase3-control-20261124-xeZzE4MM`。本报告以第二版布景及原训练第 14,388 步的权重为复核对象。北京时间 11 月 24 日 11:54，两组 3,000 步续训和两版导航评测均已结束。无记忆组通过两版门槛。空间记忆组通过正前方门槛，偏角视野内为 4/15，未达到 12/15。正式 30 格评测按预定门槛锁定。 [门槛决定](phase3-control-20261124-xeZzE4MM/repaired/gate-decision.json)

## 已验证的故障和原因

| 检查 | 实测结果 |
| --- | --- |
| 相机往返 | 冻结索引的 57,549 条训练回程动作、23 种标签，环境量化档上的 token 与环境动作不一致数均为 0。yaw 第 10 档为 0°，两端为 −10° 与 +10° |
| 复制上一步 | 验证集 2,258 / 2,401 条完全匹配（94.04%），共有 143 条动作变化；29 个带回程标签的验证回合首步，复制基线为 0/29 |
| 原无记忆轨迹复查 | 正前方 5 格 10 个回合中，9 个首步 yaw 为 +10°。4 个失败回合保持该动作 200 步。5 个成功回合先转 144 或 180 步才前移 |
| 空间记忆原版权重 | 正前方 5 格 0/10，偏角视野内 0/15、视野外 0/5。共 6,000 步均为 yaw +10°、无水平位移，死亡和位姿跳变均为 0 |

相机检查以归档中的量化动作及 MineStudio 相机配置为准。固定 pitch 为中心档时，agent 相机编号 210、220、230 分别解码为 yaw −10°、0°、+10°。此前记录的 210 到 230 是模型选择了相反的偏航方向。教师原始连续角度没有保存在冻结索引里。复制基线比两组末步的离线整条动作准确率都高，续训验证新增针对 143 条动作变化样本的精确匹配率。

[动作往返与复制基线](phase3-control-20261124-xeZzE4MM/action-audit.json) · [原无记忆逐回合首段偏航](phase3-control-20261124-xeZzE4MM/original-aligned-yaw.csv) · [空间记忆原版门槛](phase3-control-20261124-xeZzE4MM/current-spatial-nav/episodes.csv) · [空间记忆偏角门槛](phase3-control-20261124-xeZzE4MM/current-spatial-nav_offset/episodes.csv)

## 同一首帧的历史干预

使用原版失败种子 `2030990000`、同一无记忆适配器和 CPU FP32。教师与闭环首步位姿一致。两入口各构建 967 个提示 token，逐位相同。三个 RGB 帧原尺寸均为 640×360，处理为 644×364，图像网格均为 `[1,26,46]`，像素张量尺寸为 `[3588,1176]`。可见差异为像素张量的 SHA-256，训练当前帧经过 JPEG，闭环当前帧直接来自 RGB。

| 输入 | 前进 | yaw | 与教师完全匹配 |
| --- | ---: | ---: | --- |
| 闭环 RGB，原历史 | 0 | +10° | 否 |
| 同一 RGB，空历史文本 | 1 | 0° | 是 |
| 同一 RGB，历史替换为前进 | 1 | 0° | 是 |
| 同位姿的训练 JPEG，原历史 | 0 | +10° | 否 |
| 闭环 RGB 经 JPEG86，原历史 | 0 | +10° | 否 |

GPU 6 上 BF16 的 20 个偏角首步场景中，输入的文本 token、历史动作、图像网格及尺寸也完全一致，逐场差异字段均为像素张量 SHA-256。原历史、训练 JPEG 和闭环 RGB 经 JPEG86 三种条件均在 20/20 个场景输出原地右转 10°。其中教师要求右转的 10 个场景完全匹配，视野内 8/15、视野外 2/5。空历史或前进历史均在 20/20 个场景切换为直行，两种条件各为 0/20 教师动作匹配。教师在这些偏角场景的首步都要求转向。

同图像干预确认动作历史决定了这些场景是右转还是直行，JPEG 对照均保持了原生成动作。单独清空历史无法恢复视觉转向。

[对齐场景完整输入](phase3-control-20261124-xeZzE4MM/cpu-aligned-probe/seed2030990000-input-diff.json) · [对齐场景干预输出](phase3-control-20261124-xeZzE4MM/cpu-aligned-probe/start-probes.jsonl) · [20 个 BF16 干预记录](phase3-control-20261124-xeZzE4MM/input-and-history-probes/start-probes.jsonl)

## 六个逐步 token 追踪

选取原版 4 个失败和 2 个成功的场景。六个场景的首步模型原始 token ID 完全一致，均为 `[151835,151876,151907,151836]`，解码为前进关闭、pitch 0°、yaw +10°。

| 场景种子 | 原结果 | 追踪结果 | 总步数 | 连续初始右转步数 | 本步生成与历史末尾文本完全相同 |
| --- | --- | --- | ---: | ---: | ---: |
| 2030990000 | 失败 | 失败 | 200 | 200 | 199 |
| 2030990001 | 失败 | 失败 | 200 | 200 | 199 |
| 2030990002 | 失败 | 失败 | 200 | 200 | 199 |
| 2030990003 | 成功 | 成功 | 191 | 180 | 188 |
| 2030990004 | 成功 | 成功 | 155 | 144 | 152 |
| 2030990008 | 失败 | 成功 | 191 | 180 | 188 |

复跑中种子 `2030990008` 在转完 180 步后前进并抵达，原记录仍保留为失败。三个复现的失败场景中，第 1 步以后均逐字重复上一动作。三个成功追踪各只有 11 个前进步。六个追踪回合均无死亡和位姿跳变。 [逐步原始 token、历史文本及动作](phase3-control-20261124-xeZzE4MM/no-memory-step-traces) · [轨迹统计](phase3-control-20261124-xeZzE4MM/no-memory-step-traces/control-trace-summary.json)

## 修复运行

选择动作历史丢弃和 DART。从两个第 14,388 步权重继续训练。训练时按样本以 0.5 概率将全部历史动作替换为固定空动作 token，评测时总使用该 token。原始历史和 DART 的实际执行历史都保留，监督目标始终用教师对当前状态的纠正动作。DART 每个可发起扰动的回程决策以 20% 概率插入连续 1–3 步转向或停顿。四个距离档及可见导航分别选 10,000 帧。混合索引按两种来源各 6,000 条回程标签，合计持续训练 3,000 步。

DART 实机冒烟通过，产出 6 个回合和 3,881 帧。正式采集有五个分片遇到 GLFW 初始化错误，分别在独立显示号上错峰补跑成功。32 个分片最终合计 422 个回合、140,288 帧。达到全局帧配额时已冻结索引，使用当时的 415 个回合、135,727 帧，从 314 个回合选取 50,000 帧。新采集覆盖 32 个世界，冻结索引与原划分、冻结评测的场景种子和世界种子交集均为 0。 [采集全量摘要](phase3-control-20261124-xeZzE4MM/dart-collection-summary.json)

| 组别 | 冻结帧 | 返回标签 | 执行扰动 | 执行动作与纠正标签不同 |
| --- | ---: | ---: | ---: | ---: |
| 10 格 | 10,000 | 2,922 | 954 | 884 |
| 30 格 | 10,000 | 5,393 | 1,599 | 1,517 |
| 60 格 | 10,000 | 5,657 | 1,822 | 1,740 |
| 100 格 | 10,000 | 5,829 | 1,792 | 1,730 |
| 可见导航 | 10,000 | 9,510 | 2,956 | 2,857 |

最终混合索引按每组从每一来源各取 1,200 个标签，SHA-256 为 `89e1e31e455a2524402399c956b3fe076cb2b42d53f6d1cac44fe164af896da4`。混合用到的原始动作标签中，912/6,000 带偏航（15.20%）。DART 教师纠正标签中，2,563/6,000 带偏航（42.72%）。216 份归档的本地副本均通过指纹、CRC 和 SHA-256 校验。CPU 小模型验证了 LoRA 和空间模块的精确权重初始化，以及启用历史随机丢弃后的断点恢复。修复后两组导航门槛均通过方可进入正式 30 格。

[修复选择证据](phase3-control-20261124-xeZzE4MM/repair-selection.json) · [种子与数据摘要](phase3-control-20261124-xeZzE4MM/mix/summary.json) · [动作分布](phase3-control-20261124-xeZzE4MM/mix/selected-action-distribution.json) · [本地归档校验](phase3-control-20261124-xeZzE4MM/mix/archive_sha256.jsonl) · [初始化与恢复验证](phase3-control-20261124-xeZzE4MM/cpu-initialization-test.log)

### 无记忆续训验证

北京时间 05:37、06:28 和 07:17，无记忆组分别完成第 1,000、2,000 和 3,000 步的验证。验证输入固定使用训练中的统一占位 token。动作变化子集按原始未掩码的前一动作定义。实机训练计时 9,441.21 秒（2 小时 37 分 21 秒），3,000 步和 12,000 条监督标签均已写完。

| 优化步 | 回程动作 | 验证损失 | Token 准确率 | 整条动作匹配率 | 动作变化匹配 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 2,401 | 0.13100 | 95.10% | 78.84% | 57/143（39.86%） |
| 2,000 | 2,401 | 0.12020 | 96.03% | 84.84% | 59/143（41.26%） |
| 3,000 | 2,401 | 0.39379 | 89.63% | 51.40% | 62/143（43.36%） |

末步动作变化多匹配 3 条，总体完整动作匹配率比第 2,000 步低 33.44 个百分点。两版修复导航使用预定的末步适配器，末步权重与第 2,000、3,000 步恢复状态均已保存。 [续训验证 CSV](phase3-control-20261124-xeZzE4MM/repaired/train-no_memory/val_loss.csv) · [3,000 步训练日志](phase3-control-20261124-xeZzE4MM/repaired/train-no_memory/loss.csv) · [末步权重与快照](phase3-control-20261124-xeZzE4MM/repaired/train-no_memory/weight-manifest.json)

训练全程两来源各占一半。以 1,000 步为窗口，DART 的实际样本数分别为 970、2,373、2,657，每窗共 4,000 条。三窗平均训练损失分别为 0.16256、0.18973、0.19422，3,000 条损失均为有限值。末步验证波动和后段样本构成变化同时出现，现有单次续训无法归因。 [训练窗口统计](phase3-control-20261124-xeZzE4MM/repaired/train-no_memory/training-window-summary.json)

按同一训练游标重放标签顺序，三个窗口的前进标签分别为 3,324、2,966、3,017 条。无按键、yaw +10° 的原地右转标签分别为 317、550、480 条。这些是教师的监督标签。第二窗原地右转标签由 7.93% 增至 13.75%，与空间记忆权重的动作变化在时间上重合，尚无对照可证明因果。 [各优化窗口动作分布](phase3-control-20261124-xeZzE4MM/mix/update-window-action-distribution.json)

### 修复无记忆的导航门槛

| 冻结场景 | 原无记忆 | 修复后 | 恒定前进 |
| --- | ---: | ---: | ---: |
| 正前方 5 格 | 6/10 | 10/10 | 10/10 |
| 偏角视野内 | 11/15 | 15/15 | 4/15 |
| 偏角视野外观察项 | 3/5 | 5/5 | 0/5 |

修复无记忆通过 ≥8/10 和 ≥12/15 的两项门槛，偏角视野内比恒定前进多成功 11 回合。原版每个回合用时 11 步，偏角场景用时 21–172 步。 [正前方逐回合](phase3-control-20261124-xeZzE4MM/repaired/eval-no_memory-nav/episodes.csv) · [偏角逐回合](phase3-control-20261124-xeZzE4MM/repaired/eval-no_memory-nav_offset/episodes.csv)

### 空间记忆续训验证

北京时间 09:06、10:15 和 11:15，空间记忆分别完成第 1,000、2,000、3,000 步验证。三次使用相同的历史动作占位口径、6,500 帧、2,401 条动作和 11,751 个 token。

| 优化步 | 验证损失 | Token 准确率 | 整条动作匹配率 | 动作变化匹配 |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.16518 | 95.69% | 86.42% | 47/143（32.87%） |
| 2,000 | 0.23188 | 79.94% | 5.04% | 21/143（14.69%） |
| 3,000 | 0.13664 | 96.57% | 86.42% | 47/143（32.87%） |

第 2,000 步的总体完整动作匹配率下降 81.38 个百分点。2,280 条动作匹配失败，共 2,357 个错误 token，由此可知至少 2,203 条动作恰好错一个 token。已核对混合索引哈希、两个快照的优化器进度和数据游标，以及截至 2,042 步的训练损失均有限。

末步验证的整条动作和动作变化匹配率均回升至第 1,000 步水平。空间记忆训练共 12,000 条标签，耗时约 3 小时 37 分钟，三个窗口的平均训练损失为 0.22195、0.27749、0.22693。3,000 个训练步的损失均为有限值。末步验证使用固定教师前缀，实机行为以随后两版导航成绩为准。 [训练损失](phase3-control-20261124-xeZzE4MM/repaired/train-spatial/loss.csv) · [训练完成记录](phase3-control-20261124-xeZzE4MM/repaired/train-spatial/summary.json)

CPU FP32 在四个固定的短距可见验证样本上复测了最常见的前进标签。第 1,000 步在四例中均完全匹配。第 2,000 步在四例中都仅把位置 1 的前进 token `151848` 误预测为相机 token `151876`。 [CPU 逐 token 审计](phase3-control-20261124-xeZzE4MM/spatial-token-audit.json)

GPU 7 以 BF16 将此复核扩展至完整验证集。在相同的 11,751 个 token 上，首次验证与复核均正确 9,394 个，动作变化子集均为 21/143。整体精确匹配分别为 121/2,401 和 120/2,401。复核中有 2,205 条动作恰好预测错一个 token，前进 `151848` 被相机 `151876` 替换共 2,099 次，占全部 2,357 个 token 错误的 89.05%。复核使用固定目标前缀，预定自由生成闭环使用第 3,000 步权重。 [BF16 全量逐 token 审计](phase3-control-20261124-xeZzE4MM/spatial-token-audit-bf16.json)

在同一组四个需要直行的验证画面上，GPU 7 的 BF16 自由生成均正确结束。第 1,000 步四例均生成前进、yaw 为 0°。第 2,000 步四例均省掉前进并生成 yaw +10°。逐条和批量生成的 token 在八次比较中完全一致。MineStudio 解码准确执行了模型给出的动作。这个受控探针定位了两次权重之间的输出变化，具体训练因素及第 3,000 步闭环效果仍需实测。 [配对自由生成及解码记录](phase3-control-20261124-xeZzE4MM/spatial-greedy-audit.json)

将 LoRA 适配器与空间记忆模块在第 1,000 和 2,000 步之间交叉替换，继续在同一组画面上用 BF16 自由生成。两者均来自第 2,000 步时，四例均原地右转。单独换回任一第 1,000 步组件，四例均恢复前进。结果说明两个组件的联合变化足以触发该小样本故障。交叉组合只用于诊断，结论限于这四个画面。 [组件交叉探针](phase3-control-20261124-xeZzE4MM/spatial-component-audit.json)

[空间记忆续训验证 CSV](phase3-control-20261124-xeZzE4MM/repaired/train-spatial/val_loss.csv) · [实机训练配置](phase3-control-20261124-xeZzE4MM/repaired/train-spatial/training_config.json)

### 修复空间记忆的导航门槛

正前方 5 格的十个冻结场景全部成功，各用 11 步，最小距离为 2.885–2.888 格，无死亡或位姿跳变。原空间权重在相同场景为 0/10，修复后达到 ≥8/10 的要求。恒定前进同样为 10/10，按目标转向的能力还要由偏角门槛验证。 [正前方逐回合](phase3-control-20261124-xeZzE4MM/repaired/eval-spatial-nav/episodes.csv)

偏角视野内为 4/15，视野外为 0/5。16 个失败回合均耗尽步数预算，全部 20 个回合均无死亡或位姿跳变。空间记忆模型与配对恒定前进的 3,324 条 x、y、z、yaw、pitch 记录逐步相同，各回合的步数、最小距离和成功判定一致。20 个回合的朝向都未改变。 [偏角逐回合](phase3-control-20261124-xeZzE4MM/repaired/eval-spatial-nav_offset/episodes.csv) · [完整轨迹对比](phase3-control-20261124-xeZzE4MM/repaired/eval-spatial-nav_offset/full-forward-comparison.json)

## 原权重的 30 格诊断

GPU 7 使用两个 MineStudio 实例共享当前加载的模型，按两个变体分别测各 30 个冻结的 30 格场景。结果标记为 `diagnostic_d30`，原权重的导航门槛未通过。继续或止损判断使用修复后通过门槛的正式数据。

两组都完整覆盖 30 个冻结场景，均为 **0/30**。每组 30 个回合均耗尽预算，各 **22,148 步**全部为 +10° 偏航，水平位置和俯仰始终未变，死亡和位姿跳变均为 0。两组各自的 Wilson 95% 区间约为 0%–11.35%。 [原无记忆逐回合](phase3-control-20261124-xeZzE4MM/diagnostic-d30-no_memory/episodes.csv) · [原空间记忆逐回合](phase3-control-20261124-xeZzE4MM/diagnostic-d30-spatial/episodes.csv) · [原空间记忆批次计时](phase3-control-20261124-xeZzE4MM/diagnostic-d30-spatial/rollout_timing.csv)

空间记忆续训和导航改由 GPU 6 在修复无记忆门槛完成后接续。GPU 7 完成了原空间记忆诊断。修复空间记忆偏角门槛未通过，自动调度完成门槛检查并停止在正式 30 格任务之前。12:19 核查时，GPU 6 和 7 的显存与利用率均为零。 [调度记录](phase3-control-20261124-xeZzE4MM/parallel-schedule.json) · [门槛决定](phase3-control-20261124-xeZzE4MM/repaired/gate-decision.json)

## 闭环故障的层级与修复

相机往返检查和自由生成探针分别验证了解码与控制指令。原权重改变历史动作文本即可使首步从原地右转切换为前进。续训后空间记忆组在占位历史下仍会随检查点发生同样的动作切换。当前证据将复发定位至训练后的动作生成。无记忆组在偏角视野内从 11/15 提升到 15/15，恒定前进为 4/15，说明这套控制链可支持对视野内目标的转向与接近。空间记忆组当前已恢复直行，并已确定无法达到偏角门槛。正式 30 格的解锁规则要求两组都达标，空间记忆收益仍待验证。

下一轮可优先用短对照检验训练样本的时间分布。保持训练起点、标签、历史占位和门槛场景一致，让原始与 DART 样本在训练窗口内维持 1:1。在独立验证上同时记录自由生成的前进召回率、左右转向召回率和动作变化处准确率，再测冻结闭环门槛。若仍退化成直走，可单独降低空间模块学习率或做短时冻结对照。组件交叉探针限于定位问题，现行正式检查点采用第 3,000 步规则。

### 11 月 24 日下午的平衡续训

北京时间 14:25 启动第二轮实验，从两组各自原第 14,388 步适配器初始化。GPU 6 训练无记忆，GPU 7 训练空间记忆。两组用相同的冻结索引、BF16、历史动作丢弃概率 0.5、学习率 0.0001，并在每次更新中各读两条原始标签和两条 DART 标签。短试验固定 1,000 步，每 250 步保存检查点。用 CPU 小模型注入中断后，从恢复检查点得到逐参数相同的权重。 [启动配置](phase3-steering-balanced-hINMXxy1/launch-config.json) · [恢复测试](phase3-steering-balanced-hINMXxy1/cpu-recovery-test.log)

复算确认全部 1,000 个批次的来源比例以及三处检查点的后续样本顺序。此短试验检验来源交替，短索引仍沿各来源内的回合随机顺序取前缀。4,000 条标签包含 794 条可见导航和 331 条 30 格回程，前缀中 100 格为零。两组使用同一个前缀。 [训练顺序核查](phase3-steering-balanced-hINMXxy1/balanced-data-audit.json)

空间记忆第 250 步的 CPU FP32 自由生成探针从验证集可见导航中抽取直行、左转、右转各两例。六例均完整生成，其中六例均输出直行。作为诊断口径校验，上一轮已通过两版闭环门槛的无记忆第 3,000 步权重用同一抽样规则抽取每类六例，直行召回 6/6、左转 2/6、右转 0/6。单步样本反映动作偏向，冻结场景的闭环表现以完整评测为准。 [空间记忆第 250 步探针](phase3-steering-balanced-hINMXxy1/early-spatial-probe-250/steering.json) · [无记忆参照探针](phase3-steering-balanced-hINMXxy1/reference-steering-probes/no_memory.json)

空间记忆第 500 步在完全相同的六个样本上仍分别把两条直行、两条左转、两条右转预测为直行。候选训练选项可提高可见导航回程中的左右转标签权重，并按每个优化批次的权重总和归一化动作交叉熵。CPU 小模型已验证加权口径和中断后权重恢复。 [第 500 步探针](phase3-steering-balanced-hINMXxy1/early-spatial-probe-500/steering.json) · [加权恢复测试](phase3-steering-balanced-hINMXxy1/candidate-weighted-control/cpu-test.log)

上一轮空间记忆第 3,000 步权重在与无记忆参照完全相同的 18 个探针上全部输出直行。无记忆参照出现了三次主动偏航，方向与标签一致两次。 [空间记忆参照探针](phase3-steering-balanced-hINMXxy1/reference-steering-probes/spatial.json)

平衡续训的无记忆组于 15:25 完成 1,000 步全量验证。损失为 0.16332，token 准确率为 94.10%，整条动作 1,773/2,401，动作变化处 53/143。空间记忆于 15:53 完成同一验证，损失 0.16009，token 准确率 95.69%，整条动作 2,075/2,401，动作变化处 47/143。两组均已保存末步权重。无记忆组冻结正前方场景 10/10，均为 11 步。 [无记忆验证](phase3-steering-balanced-hINMXxy1/repaired/train-no_memory/val_loss.csv) · [空间记忆验证](phase3-steering-balanced-hINMXxy1/repaired/train-spatial/val_loss.csv)

平衡续训无记忆组在前八个偏角视野内场景中为 4/8。四个失败令此轮上限为 11/15。15:41 条件队列据此启动，每个变体完成平衡续训本轮两版评测后，占用对应 GPU 从各自平衡续训末步权重再训 750 步。平衡续训的最后 300 步中可见导航标签为零。加权续训因此在两来源内均匀穿插可见导航，保持原索引 20% 比例，并将可见场景的偏航标签设为 3 倍相对权重。实际训练顺序复算得到 600 条可见标签，其中左转 83、右转 77，两组相同。 [条件队列](phase3-steering-weighted-D2oX6SVa/launch-overlap.sh) · [加权续训顺序](phase3-steering-weighted-D2oX6SVa/weighted-data-audit.json)

平衡续训无记忆的完整偏角结果为视野内 8/15、视野外 0/5。20 场的第一次偏航全部向右。视野内需要右转的 8 场全部成功，需要左转的 7 场全部失败。GPU 6 在该轮结束后接续加权续训，于 16:01 完成第一步。加权无记忆第 250 步在六样本探针中正确预测两次直行和一次右转，两次应左转的样本依然预测为右转。 [无记忆偏角 CSV](phase3-steering-balanced-hINMXxy1/repaired/eval-no_memory-nav_offset/episodes.csv) · [方向诊断](phase3-steering-balanced-hINMXxy1/repaired/eval-no_memory-nav_offset/directional-audit.json) · [GPU 6 实际训练配置](phase3-steering-weighted-D2oX6SVa/repaired/train-no_memory/training_config.json) · [加权无记忆 250 步探针](phase3-steering-weighted-D2oX6SVa/early-no_memory-probe-250/steering.json)

16:31 两组平衡续训的实机评测全部结束。空间记忆正前方 10/10、偏角视野内 4/15、视野外 0/5。空间记忆的 20 条完整偏角轨迹共有 3,324 条位姿记录，与恒定前进逐步相同。正式 30 格按原门槛锁定，加权空间记忆在该组收尾后开始加载。 [平衡续训门槛决定](phase3-steering-balanced-hINMXxy1/repaired/gate-decision.json) · [完整偏角对照](phase3-steering-balanced-hINMXxy1/repaired/eval-spatial-nav_offset/full-forward-comparison.json)

对平衡空间记忆的同一末步 LoRA 做 CPU FP32 配对探针。在相同六个验证画面上，分别保留和移除记忆读取前缀。两种条件解码后都六次直行。移除前缀产生了额外的中性俯仰 token，前进和偏航动作均相同。采样范围内，推理期跳过记忆读取不足以恢复转向。 [同权重配对探针](phase3-steering-balanced-hINMXxy1/memory-read-probe/steering.json)

加权空间记忆于 16:36 完成第一步。两组训练设置核对一致，均从各自平衡版第 1,000 步初始化。加权无记忆于 16:37 完成 750 步和 3,000 条标签，16:49 完成末步验证。验证损失 0.13241，token 准确率 95.27%，整动作 1,893/2,401，动作变化 63/143。第 750 步在同六画面的 CPU 探针与第 250 步的方向预测完全相同，两次应左转均预测为右转。 [加权无记忆验证](phase3-steering-weighted-D2oX6SVa/repaired/train-no_memory/val_loss.csv) · [第 750 步探针](phase3-steering-weighted-D2oX6SVa/early-no_memory-probe-750/steering.json) · [GPU 7 实际训练配置](phase3-steering-weighted-D2oX6SVa/repaired/train-spatial/training_config.json)

加权无记忆正前方为 1/10。失败的九个回合共发生 235 次右转，左转零次，均无死亡或位姿跳变。唯一成功的回合沿恒定前进轨迹用 11 步到达。加权空间记忆第 250 步在同一组六个验证画面上全部生成右转，该组平衡版前两个检查点在六例全部生成直行。 [加权无记忆正前方方向统计](phase3-steering-weighted-D2oX6SVa/repaired/eval-no_memory-nav/directional-audit.json) · [加权空间记忆 250 步探针](phase3-steering-weighted-D2oX6SVa/early-spatial-probe-250/steering.json)

下一轮采集匹配起点的三种相机朝向。冒烟的两个位置均通过几何和标签校验，正前方的目标像素横向中心约为 0.501，左侧 0.263–0.304，右侧 0.697–0.734。正式采集使用 32 路 CPU 和 128 个新位置。两组计划从各自加权前的平衡权重初始化，两轮共 256 次更新，每步使用同位置三条配对动作和三条旧回放动作。 [配对采集冒烟](phase3-steering-paired-2AGguG2e/smoke-index/summary.json) · [配对回放启动脚本](phase3-steering-paired-2AGguG2e/launch-train.sh)

完整采集含 128 个位置、384 个场景、1,152 帧。第 22 路原世界的地形校验连续失败，替换为 `21170222` 后补齐，另外 31 路保留。新场景和世界种子与已有数据及冻结评测零交集。全图最大色块在六个画面误选背景，改用已验证的金柱投影范围后，全部 384 帧的目标范围内都有金色像素，最少 2,420 像素。 [补采脚本](phase3-steering-paired-2AGguG2e/recover-capture.sh) · [采集索引摘要](phase3-steering-paired-2AGguG2e/index/summary.json) · [配对回放索引](phase3-steering-paired-2AGguG2e/replay-index/summary.json)

配对回放两轮共 1,536 条监督标签，配对数据 768 条，原始与 DART 各 384 条。直行、左转、右转在配对数据中各 256 条。第 64、128、192 步的样本顺序支持精确恢复，CPU 小模型中断测试逐参数匹配。两组继续用原有 6,500 帧验证集和冻结门槛。 [配对训练回放核查](phase3-steering-paired-2AGguG2e/replay-order-audit.json) · [CPU 恢复测试](phase3-steering-paired-2AGguG2e/cpu-training-test.log)

加权无记忆全部 20 个偏角场景均失败，4,000 个控制步中 501 步右转、0 步左转，均无死亡或位姿跳变。空间记忆加权末步全量验证损失 0.17357，整动作 2,075/2,401、动作变化处 47/143。空间记忆第 750 步在同六个画面上再次全部直行，第 250 步在相同画面全部右转。GPU 6 在 17:52 收齐加权结果后启动配对续训。 [加权无记忆偏角方向](phase3-steering-weighted-D2oX6SVa/repaired/eval-no_memory-nav_offset/directional-audit.json) · [加权空间验证](phase3-steering-weighted-D2oX6SVa/repaired/train-spatial/val_loss.csv) · [空间末步方向探针](phase3-steering-weighted-D2oX6SVa/early-spatial-probe-750/steering.json)

18:29 两组加权评测结束。空间记忆正前方 10/10、偏角视野内 4/15、视野外 0/5，全部 3,324 条偏角位姿记录与恒定前进相同。正式 30 格锁定。GPU 7 在自身加权评测退出后开始配对续训。 [加权门槛决定](phase3-steering-weighted-D2oX6SVa/repaired/gate-decision.json) · [加权空间偏角轨迹核查](phase3-steering-weighted-D2oX6SVa/repaired/eval-spatial-nav_offset/full-forward-comparison.json)

配对无记忆在 18:15 完成 256 步，18:27 左右完成 6,500 帧验证。验证损失 0.39652，token 准确率 91.03%，整动作 1,398/2,401，动作变化 62/143。固定六画面探针的末步完整生成正确预测两次直行、一次右转，两次应左转分别预测为直行和右转。闭环原始动作记录用于进一步检验方向。 [配对无记忆验证](phase3-steering-paired-2AGguG2e/repaired/train-no_memory/val_loss.csv) · [配对无记忆 256 步探针](phase3-steering-paired-2AGguG2e/early-no_memory-probe-256/steering.json)

配对无记忆正前方最终 3/10。1,757 步均完整解码，解码和实机偏航逐步相同，死亡和位姿跳变均为零。目标偏角在教师前进阈值 ±8° 内时，模型前进 38 次，原地转向 85 次。整组共右转 1,086 次、左转 633 次。配对监督让模型在实机输出了两种方向，同时仍会在该前进时原地来回转。 [配对无记忆正前方原始动作统计](phase3-steering-paired-2AGguG2e/repaired/eval-no_memory-nav/directional-audit.json)

配对无记忆末步在固定六个验证画面上额外计算了直行、最大左转、最大右转的完整 token 序列对数概率。三条候选中的最高分动作每次都等于原贪心输出，方向匹配均为 3/6。两次应左转的画面分别预测成右转和直行。其中一次生成右转前，无前进相机分支约为 99.8%，前进分支约为 0.1%。这一小样本没有观察到完整动作打分改善方向选择。 [完整动作概率对照](phase3-steering-paired-2AGguG2e/complete-action-probe/steering.json)

将首轮已通过导航门槛的无记忆第 3,000 步 LoRA 接上同轮已训练的空间记忆模块。在 CPU FP32 的 18 个近距验证画面上，空间读取开启、临时移除读取和原无记忆模型三种条件生成的 token 序列 18/18 完全一致。近距小样本中的控制输出保持一致，已排队测两场正前方、四场偏角实机诊断。 [控制权重组合与逐 token 数据](phase3-steering-paired-2AGguG2e/shared-adapter-probe/steering.json) · [六场实机诊断入口](phase3-steering-paired-2AGguG2e/shared-adapter-probe/navigation-diagnostic.sh)

19:22 配对无记忆完成 30 个冻结场景，正前方 3/10，偏角视野内 0/15，视野外 1/5。共 5,705 个动作全部完整解码，偏航与实机逐步一致。整个偏角组在目标落在 ±8° 时前进 120 次，原地转向 543 次。唯一的偏角成功为视野外种子 2030990116，148 步到达，恒定前进同种子失败。 [偏角完整结果](phase3-steering-paired-2AGguG2e/repaired/eval-no_memory-nav_offset/episodes.csv) · [偏角原始动作统计](phase3-steering-paired-2AGguG2e/repaired/eval-no_memory-nav_offset/directional-audit.json)

19:32 配对空间记忆完成 256 步和固定验证，整动作 177/2,401（7.37%），动作变化处 24/143，token 准确率 80.46%。同一时段，控制 LoRA 接空间记忆的实机诊断正前方 2/2，偏角 3/4。四个偏角回合的第一个非零偏航均向左，成功步数 65、89、137。19:35 开始补跑其余冻结场景，之后按原门槛核验 10 个对齐场景及 20 个偏角场景。 [配对空间验证](phase3-steering-paired-2AGguG2e/repaired/train-spatial/val_loss.csv) · [权重组合偏角诊断](phase3-steering-paired-2AGguG2e/shared-adapter-probe/nav_offset/episodes.csv) · [完整门槛验证入口](phase3-steering-paired-2AGguG2e/shared-adapter-probe/confirm.sh)

19:57 共享 LoRA 组合通过完整导航门槛。对齐 10/10，偏角视野内 14/15，视野外 5/5。全部 2,302 步完整解码并与实机吻合，其中前进 624 步，左转 1,678 步，右转为零。20 个偏角场景的第一步有 19 个向左，导航成功包含单向搜转后前进。正式 30 格采用同一个无记忆 LoRA，空间组额外读取已训练的空间模块。GPU 6 于 19:57 开始无记忆正式评测，空间组在 GPU 7 配对评测结束后接续。 [共享权重完整导航判定](phase3-steering-paired-2AGguG2e/shared-adapter-probe/confirm-verdict.json)

20:32 配对续训的空间组评测正常退出，正前方 0/10，偏角视野内 0/15，视野外 0/5。逐步核对的 6,000 个动作全部原地向左转 10°，无死亡和位姿跳变，该配对版本未通过正式门槛。GPU 7 随后开始独立的共享 LoRA 空间组评测。共享 LoRA 的无记忆组截至 21:00 为 0/6，最近距离 23.36–29.98 格。共享 LoRA 对照控制了 LoRA 权重，空间组仍同时改变了读取前缀和输入位置索引。 [配对门槛判定](phase3-steering-paired-2AGguG2e/repaired/gate-decision.json) · [共享 LoRA 无记忆逐回合结果](phase3-steering-paired-2AGguG2e/shared-adapter-probe/d30/no_memory/episodes.csv)

21:05 空间组的首批三个正式种子均失败。与无记忆组相同种子的最近距离分别为 29.71/29.31、29.98/29.98、29.68/23.36 格。空间组 2,210 步中有 2,196 步实测偏航为 −10°，三场累计路径长为 1.53、0、1.27 格。无记忆组同三场累计路径长为 10.69、1.48、10.34 格。两组均无死亡或位姿跳变。该批主要观测到持续转向，两个完整 30 种子阶段仍在运行。 [共享 LoRA 空间组逐回合结果](phase3-steering-paired-2AGguG2e/shared-adapter-probe/d30/spatial/episodes.csv) · [首批时长](phase3-steering-paired-2AGguG2e/shared-adapter-probe/d30/spatial/rollout_timing.csv)

21:12 核验无记忆前三批为 0/9，空间组首批为 0/3。空间组首批含模拟器布景和预处理共 1,821 秒，聚合 1.21 FPS，当前批次之后继续按原定顺序运行。 [无记忆计时与进度](phase3-steering-paired-2AGguG2e/shared-adapter-probe/d30/no_memory/rollout_timing.csv)

## 空间网格保留与记忆利用

用 120 条与第二版冻结场景配置完全一致的外出轨迹重放网格。CPU 顺序回放的保留索引逐例匹配模型原网格实现，共核对 38,245 个仍保留的观测。当前 64 格宽的网格在 10、30、60 格各 30/30 保留完整起始停留段，100 格为 0/30。网格宽度为 128 或 256 格时，100 格的 30/30 保留完整起始停留段。10、30、60 格当前网格中每回合还有 1–5 条后续观测写入同一个起始槽，中位数为 3。这里统计的是槽和写入位置，学习后的视觉分量权重由干预实验另行测量。 [冻结轨迹逐回合网格核查](phase3-memory-20261124/grid-retention/episodes.csv) · [网格汇总](phase3-memory-20261124/grid-retention/summary.json) · [逐索引复核](phase3-memory-20261124/verified-grid.json)

120 个冻结回合的第一条记录与金柱水平距离均为 2.828 格，落在 3 格成功范围内。CPU 实机分别测试回到第一个历史位姿和沿记录的位姿逆序折返，动作选择使用当前和已存位置，金柱坐标用于成功判定。该对照测量没有视觉目标识别时可达到的成绩。 [位姿控制启动脚本](phase3-memory-20261124/pose-controls/launch.sh)

### 仅凭位姿的折返

| 末段控制 | 距离 | 直奔起始位姿 | 沿历史位姿折返 |
| --- | --- | --- | --- |
| 0.4 格容差 | 30 格 | 0/30 | 0/30 |
| 0.4 格容差 | 100 格 | 0/30 | 4/30 |
| 0.1 格容差，末段减速 | 30 格试跑 | 0/3 | 3/3 |
| 0.1 格容差，末段减速 | 100 格试跑 | 0/3 | 3/3 |
| 0.1 格容差，末段减速 | 30 格冻结复测 | 0/30 | 30/30 |
| 0.1 格容差，末段减速 | 100 格冻结复测 | 0/30 | 30/30 |

直奔策略在 30 格档最多接近到 20.81 格，在 100 格档最多接近到 90.80 格。原折返策略在 30 格档的 30 个失败回合都停在离金柱 3.029–3.046 格。全部已完成的位姿对照均无死亡和位姿跳变。 [完整粗容差逐回合数据](phase3-memory-20261124/coarse-merged/episodes.csv) · [精确容差试跑](phase3-memory-20261124/precise-smoke/episodes.csv)

粗容差运行中，100 格种子 `2026112126` 两次无法实时完成教师外出段。使用场景配置逐字段匹配的历史外出记录与冻结回程起点补齐了同一试次，并保留原失败记录。该种子在粗容差下停在 3.019 格，在同一历史轨迹上收紧容差后达到 2.911 格。精确容差全量复测使用档案中的冻结历史。 [实时分片错误](phase3-memory-20261124/pose-controls/retry/worker-021/pose-controls.csv) · [同种子补跑](phase3-memory-20261124/pose-controls-archived/coarse-recovery/worker-021/pose-controls.csv) · [同种子精确试跑](phase3-memory-20261124/pose-controls-archived/smoke/worker-021/pose-controls.csv)

20:55 精确容差冻结复测落盘，两档各 30 个固定种子均成功折返，Wilson 95% 区间均为 88.6%–100%。平均控制步数为 260.1 和 585.8，死亡和位姿跳变均为零。30 格种子 `2026112117` 的第一次布景检查失败，保留该次记录并以同一个种子补跑成功。该对照给出了真值位姿和历史路线充分时的可执行解。 [冻结复测逐回合结果](phase3-memory-20261124/pose-controls-archived/episodes.csv) · [补跑原始结果](phase3-memory-20261124/pose-controls-archived/retry/worker-012/pose-controls.csv)

### 空间读取结构的定位

现有读取器将相对位姿编码加到注意力的键，值来自视觉槽。把所有历史视觉特征清零后，投影得到相同的视觉值；即使注意力权重因位姿而变，输出的加权平均仍接近常数。在五条真实 30 格轨迹上，固定当前位姿并旋转历史 90°，原权重的前缀相对变化约为 `7.6e-8` 至 `8.2e-8`。临时把 `0.5` 倍位置嵌入也加到值里，变化增至 `0.000247–0.000771`。这里验证了位姿信号进入读出值的通路，行为效果仍需用真实图像与闭环检查。 [位置读取代码](phase3-memory-20261124/position-read-probe.py) · [五例前缀数值](phase3-memory-20261124/position-read-probe/position-read.json)

在一条 30 格留出轨迹的回程起点，每 16 帧取一条并保留近期帧，从 336 条历史中写入 23 条。真实历史、打乱视觉、清零视觉、将历史位姿旋转 90°、移除前缀五种条件的生成 token 完全一致，均为左转。教师也是左转，角度档位不同。真实记忆下教师整动作的对数概率为 `−0.774`，移除前缀为 `−0.812`。这是单个 CPU FP32 输入的稀疏记忆证据，完整历史另评。 [稀疏 30 格逐条件结果](phase3-memory-20261124/sparse-keyframes/d30.json)

100 格的稀疏探针从 666 条历史中写入 44 条。64 格宽网格保留 23 条，其中起始关键帧为 0/3。128 格宽网格保留全部 44 条，起始关键帧为 3/3。前缀相对变化约 `0.000859`，两个条件生成相同左转动作，教师动作对数概率分别为 `−0.8371` 和 `−0.8373`。这次扩容恢复了观测，首个动作保持相同。 [稀疏 100 格逐条件结果](phase3-memory-20261124/sparse-keyframes/d100.json)

在相同 30 格留出输入，把三张起始关键帧的金柱区域裁切并放大，仅替换这三张帧的空间写入特征。读取前缀相对变化约 `1.7e-5`，生成动作和原历史相同，教师动作对数概率的差约 `2.0e-6`。该实验只改历史特征，近期观察、位姿、LoRA 和其它写入项全部相同。 [显著目标关键帧干预](phase3-memory-20261124/focused-history/d30.json)

训练入口已支持加载并冻结一份独立通过导航门槛的无记忆 LoRA，只更新空间模块；同时支持把位姿编码加入读取值。三步 CPU 小模型冒烟后，28 个 LoRA 张量逐值相同，19 个空间模块张量发生更新，评测入口逐值恢复了空间权重和读取配置。原训练的旧配置、恢复、微批及读超时回退测试也通过。实机 7B 尚未运行这一训练配置。 [冻结控制器冒烟](phase3-memory-20261124/frozen-controller-smoke/summary.json) · [原训练回归](phase3-memory-20261124/frozen-controller-smoke/regression.log)

### 用历史 RGB 选择折返目标

从第二版冻结索引的验证与测试划分取全部 14 个 30/60/100 格回合，分别为 6/4/4。每隔八帧查看历史 RGB，选择最大金色连通区域所在帧的记录位姿。使用完整历史和跳过起始停留后均为 14/14 落在金柱 3 格内。跳过停留时，14 个回合全部选中仍处于起点的第 40 帧。打乱 RGB 与位姿的对应关系后为 3/14，按三档分别为 2/6、0/4、1/4。直接选第一条位姿仍为 14/14。这项离线诊断用真实历史图像识别工作位置，同时表明本布景的视觉检索收益需要与存储起点的对照配套评估。已安排把四种选择接入同一折返控制器，在同一批留出场景上从冻结回程位姿开始闭环。 [检索逐回合数据](phase3-memory-20261124/rgb-anchor/heldout/anchors.csv) · [检索汇总](phase3-memory-20261124/rgb-anchor/heldout/summary.json)

21:06 四种选择在同一批留出场景上完成闭环，共 56 个回合，无死亡或位姿跳变。正确 RGB 检索、跳过起始停留的 RGB 检索及第一条历史位姿均为 14/14，打乱关联为 3/14。实际用于控制的每一个锚点都与离线检索结果逐例一致。这个原型依赖真值位姿和金色检测，提供可复核的任务对照。 [RGB 与折返闭环 CSV](phase3-memory-20261124/rgb-pose-replay/episodes.csv) · [分档及平均步数](phase3-memory-20261124/rgb-pose-replay/summary.json)

### 后续训练和任务设计

优先让空间读取直接输出可执行的相对路点，由同一控制器完成跟踪。可以先在已冻结的导航 LoRA 上训练空间读取，监督目标用教师的下一个路点和返回动作；对照实验保持执行器一致。100 格的目标历史需要单独保留关键帧或扩展物理范围，闭环验证仍需检查恢复的历史是否实际改变了行动。现有 `t1_rgb_pose_replay.py` 将历史图像检索出的观察位置与记录的折返路线相连，提供学习模型的可运行对照。

若论文主张视觉内容提供额外收益，建议独立版本将出生点设在所有目标成功区域外，并让匹配路线先后经过两个外形相同的遮挡地标。成对场景交换金柱与干扰柱，保持同一路线、起点和终点观测。记录视觉加位姿、打乱关联、只保留位姿及无记忆策略。此设计可以检查控制正确时是否真正选择了历史里出现过的目标，也应核查无记忆依次搜索是否能在预算内通过。当前冻结 T1 仍保留原场景和判定，用于固定 LoRA 的比较。
