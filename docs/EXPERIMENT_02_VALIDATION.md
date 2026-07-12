# 实验 2 正式实验前加固验收报告

本报告记录“经典运动模型与确定性 GRU 轨迹预测基线”的正式实验前加固和真实 smoke 验证。所有指标来自 `outputs/dataset_v2` 的 smoke 数据，只用于工程验收，不构成论文正式结论。本轮未实现概率 GRU、Deep Ensemble、强化学习、世界模型、ROS 或控制器，也未生成或运行正式大规模数据集。

## 1. 最终验证环境

最终重跑使用的解释器为：

`/tmp/thesis-experiment-modern/bin/python`

| 组件 | 实测版本 |
|---|---:|
| Python | 3.10.4 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| Pandas | 2.3.3 |
| Matplotlib | 3.10.8 |
| PyYAML | 6.0.3 |
| PyTorch | 2.5.1+cu121 |
| pytest | 9.1.1 |
| CUDA 可用 | 否 |

本轮还曾在 Python 3.11.0、NumPy 2.4.6、PyTorch 2.13.0+cpu 环境完成四条训练/评价命令和当时的完整测试；`run.log` 保留了 PyTorch 2.13 的实际执行记录。最终代码在上表的 Python 3.10 环境裸跑完整测试，结果为 182 项测试通过。

正式实验推荐 Python 3.10 或 3.11。Python 3.7 仅作为历史兼容记录，不再是正式实验推荐环境。本轮没有 Python 3.13 或 CUDA 硬件实机验证，不能声称已在这两种环境测试。

## 2. 实际执行命令与退出状态

为避免依赖当前 shell 的旧 Python 3.7，以下命令实际以临时现代解释器执行。没有在命令前设置任何 PyTorch、OpenMP、MKL 或 OpenBLAS 线程环境变量。

| 实际命令 | 退出状态 | 结果 |
|---|---:|---|
| `/tmp/thesis-experiment-modern/bin/python scripts/train_prediction_baseline.py --config configs/experiment_02_smoke.yaml` | 0 | 单 seed 训练完成 |
| `/tmp/thesis-experiment-modern/bin/python scripts/evaluate_prediction_baselines.py --config configs/experiment_02_smoke.yaml` | 0 | 四种方法评价完成，258 个 test window |
| `/tmp/thesis-experiment-modern/bin/python scripts/train_prediction_baseline.py --config configs/experiment_02_multiseed_smoke.yaml` | 0 | 三 seed 训练完成 |
| `/tmp/thesis-experiment-modern/bin/python scripts/evaluate_prediction_baselines.py --config configs/experiment_02_multiseed_smoke.yaml` | 0 | 三个 GRU seed 全部评价完成 |
| `/tmp/thesis-experiment-modern/bin/python -m pytest -q` | 0 | **182 passed in 4.90s** |
| `git diff --check` | 0 | 无空白错误 |
| formal 配置加载检查 | 0 | seeds、4/4 线程和 scene 统计单位解析正确，未运行正式实验 |

最后运行的是 multi-seed 配置，因此 `outputs/experiment_02` 中同名汇总文件对应三 seed smoke 结果。

## 3. 数据、输入白名单与泄漏边界

使用同一份 `outputs/dataset_v2/test.npz`：

| split | window | episode | scene |
|---|---:|---:|---:|
| train | 753 | 90 | 30 |
| validation | 249 | 30 | 10 |
| test | 258 | 30 | 10 |

所有神经网络输入只能由 `PredictionDataset` 输出：

- `history_position`
- `history_velocity`
- `history_mask`
- `history_velocity_mask`
- `time_step_seconds`

唯一监督标签是 `future_position`。`history_true_position`、`trajectory_type`、`coordinate_origin`、完整 episode 真值及所有未来运动参数均不得进入模型。`trajectory_type` 和 `occlusion_length_bin` 只在全部模型推理结束后加载，用于分组评价和绘图。

最终 `predictions.npz` 包含 3 个经典基线结果和 3 个真实 GRU seed 结果，预测 shape 为 `(6, 258, 20, 2)`；经典基线 seed 使用 `-1` sentinel，CSV 中保持空值，没有人为复制成三份。

## 4. 本轮加固结果

### 4.1 现代 NumPy mask 测试

无效 mask 回归测试现在显式构造 `float32` 的 `-1.0` 数组，不再向 `uint8` 写入负数。该测试真实进入 `build_gru_features` 并验证抛出 `ValueError`。

### 4.2 checkpoint 安全加载与恢复

- `best.pt`、`last.pt` 和 resume 状态仅保存 Tensor 及安全的标量/容器；
- NumPy MT19937 key 数组转换为 Tensor，Python 与 NumPy RNG 状态使用显式安全结构；
- CUDA 可用时保存全部 CUDA RNG state，以支持含 dropout 的正式配置恢复；
- 加载始终显式调用 `torch.load(..., weights_only=True)`；
- 安全加载失败不会回退到 `weights_only=False` 或任意 pickle；
- 过旧且不支持 `weights_only` 的 PyTorch 会明确报错；
- active resume checkpoint 的严格加载测试通过；
- 普通中断恢复与连续训练逐位一致；
- 新增 early-stopping 终止 epoch 后中断的回归测试，恢复不会额外训练一轮，best/last 权重逐位一致。

最终六个 seed 专属模型文件均通过直接 `torch.load(path, weights_only=True)` 检查：

- `checkpoints/seed_20260201/best.pt`、`last.pt`
- `checkpoints/seed_20260202/best.pt`、`last.pt`
- `checkpoints/seed_20260203/best.pt`、`last.pt`

### 4.3 PyTorch 线程

训练和评价在开始计算前显式应用 YAML 中的 intra-op / inter-op 设置并写入 `run.log`：

- single-seed smoke：1 / 1；
- multi-seed smoke：1 / 1；
- formal 配置：4 / 4，可按机器修改。

fresh subprocess 回归测试使用非默认值 2 / 2，并通过 PyTorch getter 确认配置确实生效，不依赖 `OMP_NUM_THREADS`。

测试自身的线程限制与训练配置相互独立：`tests/conftest.py` 在收集测试模块前把 `OMP_NUM_THREADS`、`MKL_NUM_THREADS` 和 `OPENBLAS_NUM_THREADS` 设为 1，并调用 PyTorch setter 固定测试主进程为 intra-op/inter-op 1/1。用户直接执行裸 `python -m pytest -q` 即可；该设置不会修改训练或 formal YAML，也不妨碍 fresh subprocess 显式验证 2/2。

### 4.4 配置参数实际生效

- `config.fill_value` 被显式传给训练、验证和评价的 `PredictionDataset`；
- 非默认 `fill_value=-17.25` 的训练加载测试和 `fill_value=-23.5` 的评价加载测试均确认填充值变化；
- `config.normalization_epsilon` 被传给 `PredictionNormalizer.fit(..., minimum_scale=...)`；非默认 `0.75` 测试确认统计 scale 发生变化；
- `config.evaluation.default_statistical_unit` 实际控制 ADE/FDE 主图取 episode 或 scene 表；非默认 scene 选择测试通过；
- smoke 实际日志显示主图使用 episode，formal 配置解析结果为 scene。

## 5. 多 seed 训练结果

三 seed smoke 使用 hidden size 16、单层 GRU、3 epoch，每个模型 1,880 个参数。该配置只用于验证多 seed 管线。

| seed | best epoch | validation loss | training time (s) | 参数量 | checkpoint |
|---:|---:|---:|---:|---:|---|
| 20260201 | 3 | 0.4648387832 | 0.1027 | 1,880 | `checkpoints/seed_20260201/{best,last}.pt` |
| 20260202 | 3 | 0.5079658661 | 0.1036 | 1,880 | `checkpoints/seed_20260202/{best,last}.pt` |
| 20260203 | 3 | 0.4933633286 | 0.1080 | 1,880 | `checkpoints/seed_20260203/{best,last}.pt` |

顶层 `checkpoints/best.pt` 和 `last.pt` 只作为部署便利文件，不替代三个 seed 的独立论文统计。训练没有根据 validation 最优 seed 丢弃其他 seed。

## 6. 多 seed 评价结果

### 6.1 GRU 逐 seed

| seed | ADE episode | ADE scene | FDE episode | FDE scene |
|---:|---:|---:|---:|---:|
| 20260201 | 0.4168381016 | 0.4131364984 | 0.8427475528 | 0.8390808007 |
| 20260202 | 0.4422621532 | 0.4387058292 | 0.8992163245 | 0.8949478308 |
| 20260203 | 0.4278141108 | 0.4245982508 | 0.8297031721 | 0.8261563581 |

### 6.2 mean ± std

正式多 seed 结果采用独立 seed 的样本标准差（`ddof=1`）。它衡量独立训练 seed 的运行间波动，不是 window 之间的标准差；禁止用高度重叠 window 的离散程度冒充随机种子不确定性。经典确定性基线只有一个结果，按规则报告 `seed_count=1`、`std=0`。

| 指标 | window | episode | scene |
|---|---:|---:|---:|
| GRU ADE | 0.4239003879 ± 0.0129948825 | 0.4289714552 ± 0.0127514777 | 0.4254801928 ± 0.0128074601 |
| GRU FDE | 0.8503103952 ± 0.0371950205 | 0.8572223498 ± 0.0369480635 | 0.8533949965 ± 0.0365614391 |

从 `summary_metrics_by_seed.csv` 独立重算全部 24 行（多 seed 使用 `ddof=1`，单 seed 按规则取 0）：最大 mean 差异为 **0.0**，最大 std 差异为 **0.0**。

### 6.3 经典基线和 GRU 正式统计单位对比

经典方法只评价一次，因此 `seed_count=1`、`std=0`。

| 方法 | ADE episode | ADE scene | FDE episode | FDE scene |
|---|---:|---:|---:|---:|
| Constant Position | 0.5224862960 | 0.5193801564 | 0.9801074445 | 0.9767781842 |
| Constant Velocity | 0.6556229205 | 0.6596415448 | 1.1955498905 | 1.2059774294 |
| CV Kalman Filter | 0.1586459933 | 0.1595176916 | 0.2963526938 | 0.2993731241 |
| Deterministic GRU | 0.4289714552 ± 0.0127514777 | 0.4254801928 ± 0.0128074601 | 0.8572223498 ± 0.0369480635 | 0.8533949965 ± 0.0365614391 |

这些数值只属于 3 epoch 的 multi-seed smoke。重叠 window 不是独立实验样本；论文结论必须使用 episode 或 scene 等权聚合。

### 6.4 分组结果

GRU 的 episode-level ADE 三 seed mean ± std：

| 运动类型 | ADE |
|---|---:|
| constant_velocity | 0.351717 ± 0.025182 |
| constant_acceleration | 0.566467 ± 0.017439 |
| constant_turn | 0.467609 ± 0.017351 |
| stop_and_go | 0.341689 ± 0.008692 |
| piecewise_direction | 0.417376 ± 0.020897 |

遮挡分组属于 window-level 描述性统计；五个 bin 均有结果，但 `11-15` 和 `16-20` 分别只有 5 和 3 个 window，不足以形成论文结论。per-horizon、运动类型和遮挡图的 seed 误差条均调用同一 `ddof=1` 规则，并直接读取已保存的逐 seed CSV。完整数据见相应 CSV。

## 7. 运行时间

最终运行设备为 CPU。单样本与 batch 的每 window 时间（ms）：

| 方法 | 单样本 | batch |
|---|---:|---:|
| Constant Position | 0.0532 | 0.00145 |
| Constant Velocity | 0.0394 | 0.00556 |
| CV Kalman Filter | 0.3334 | 0.26153 |
| GRU seed 20260201 | 0.6569 | 0.00891 |
| GRU seed 20260202 | 0.6923 | 0.01002 |
| GRU seed 20260203 | 0.5820 | 0.01120 |

机器无可用 CUDA，因此没有 GPU 实测时间。

## 8. 输出文件

最终 multi-seed 产物：

| 文件 | 字节数 |
|---|---:|
| `config.yaml` | 968 |
| `normalization.json` | 419 |
| `training_history.csv` | 969 |
| `seed_training_summary.csv` | 487 |
| `per_window_metrics_by_seed.csv` | 147,261 |
| `per_episode_metrics_by_seed.csv` | 26,408 |
| `per_scene_metrics_by_seed.csv` | 6,650 |
| `per_horizon_metrics_by_seed.csv` | 7,860 |
| `summary_metrics_by_seed.csv` | 2,118 |
| `summary_metrics_mean_std.csv` | 1,465 |
| `predictions.npz` | 339,340 |
| `runtime.csv` | 1,062 |
| `run.log` | 31,856 |
| 每个 seed 的 `best.pt` / `last.pt` | 各 10,730 |
| 顶层 `best.pt` / `last.pt` | 各 10,730 |
| `resume_state.pt` | 3,700 |
| `figures/` | 8 张 PNG，共 694,319 |

逐层级无 `_by_seed` 的兼容 CSV 同样保留 seed 字段。最终行数为：window 1,548、episode 360、scene 120、horizon 120、summary-by-seed 36。

## 9. 测试覆盖

最终裸 `pytest`：**182 passed in 4.90s，0 failed，0 skipped**。命令没有外部线程前缀。

本轮重点回归覆盖：

- 浮点无效 mask 真正触发输入校验；
- best/last 直接 `weights_only=True` 加载；
- 含 active state、优化器和 RNG 的 resume checkpoint 严格加载；
- 中断恢复与连续训练逐位一致；
- early-stopping 终止快照恢复不多训练；
- 非默认 2/2 线程实际应用；
- pytest 主进程在收集早期固定为 1/1，三个本地线程环境变量均为 1；
- 非默认 fill、normalization epsilon 和统计单位实际改变程序行为；
- 三个 seed checkpoint 与训练摘要完整；
- 经典基线不被复制，GRU seed 身份保留；
- seed 样本标准差与 `ddof=1` 独立重算一致，单 seed 标准差为 0；
- predictions 的 ID 和原始 test 顺序一致。

## 10. 已发现并修复的问题

1. `uint8` mask 写入 `-1` 在 NumPy 2.x 下测试构造失败：改用显式 `float32` 无效 mask。
2. resume checkpoint 直接包含 NumPy RNG ndarray：改为 Tensor 和安全基本容器。
3. checkpoint 加载存在旧版非安全 fallback：移除 fallback，统一强制 `weights_only=True`。
4. PyTorch 线程仅依赖环境默认：加入 YAML 显式配置、实际应用和日志。
5. 多 seed 只在配置层存在：完成逐 seed 训练、checkpoint、评价、CSV、mean/std 和误差条图。
6. `fill_value`、`normalization_epsilon`、`default_statistical_unit` 曾只读取未完整验证：生产路径显式传递并新增非默认值测试。
7. early-stopping 终止 epoch 的中断恢复可能额外训练一轮：恢复入口识别终止计数并直接返回。
8. CUDA dropout 恢复缺少 CUDA RNG：checkpoint 现在保存和恢复全部 CUDA RNG state。
9. 高线程机器裸跑小型 GRU 测试严重变慢：`tests/conftest.py` 在测试收集早期固定测试进程为 1/1，不改变训练配置。
10. 多 seed 误差条曾使用 population std：总汇总、per-horizon、运动类型和遮挡分组统一改为独立 seed 样本标准差 `ddof=1`；运行时间重复测量仍保留原定义并在代码中明确区分。

## 11. 尚未解决的问题

- `outputs/dataset_v2_formal` 不存在，正式 Dataset V2 尚未生成；
- `configs/experiment_02_formal.yaml` 只完成加载检查，未运行正式训练或评价；
- 未在 Python 3.13 实机运行最终代码；
- 无 CUDA 硬件，GPU 训练、运行时间和 CUDA 中断逐位恢复尚未实机验收；
- smoke 数据为二维合成小数据，三 seed 各仅训练 3 epoch，不能作为论文结论；
- single-seed 与 multi-seed smoke 共用输出目录，后运行配置会覆盖同名汇总文件，运行时必须保留配置副本和日志。

## 12. 正式实验边界与结论

`configs/experiment_02_formal.yaml` 已真实加载：3 个 seed、4/4 PyTorch 线程、hidden size 128、2 层 GRU、scene-level 默认统计。配置指向的 `outputs/dataset_v2_formal` 明确未生成，本轮也未运行它。

结论：训练、恢复、安全加载、多 seed 评价和分层统计框架已具备正式实验所需的软件边界；进入正式训练前仍必须先生成并独立验收正式 Dataset V2，并根据实际 CPU/GPU 机器确认线程数。当前所有性能数值均为 smoke 工程结果。
