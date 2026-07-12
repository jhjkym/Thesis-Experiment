# Thesis-Experiment

## 当前研究方向

面向复杂林下环境的四旋翼无人机感知、规划与控制研究。

本仓库当前完成了无泄漏数据生成以及确定性轨迹预测基线，不包含概率预测、规划训练或无人机控制模型。

## 当前已完成实验

实验 1：林下动态目标遮挡场景与观测数据生成验证。

实验 1.1：多场景、多运动模式、按 scene 隔离的数据集构建与验证。

实验 2：经典运动模型与确定性 GRU 轨迹预测 smoke 基线。

目前已完成：

- 二维林下场景生成；
- 圆形树干建模；
- 匀速动态目标轨迹；
- 传感器视线遮挡判断；
- 观测噪声和随机丢帧；
- 历史与未来轨迹窗口；
- 局部坐标转换；
- 数据保存和可视化；
- 单元测试。
- 30/10/10 个 train/validation/test scene 的独立 seed 生成；
- scene、episode、window 三级数据组织；
- 匀速、匀加速、匀速转弯、启停和分段转向五种运动；
- scene 级无泄漏划分和独立验证程序；
- 从保存 NPZ 读取的九张数据集可视化。
- 预测数据加载白名单、缺失值有限填充和速度有效性 mask；
- 可见历史门槛、不可观测 episode 拒绝重采样和传感器树干净距；
- window、episode、scene、运动类型和遮挡分组的通用指标聚合。
- Constant Position、Constant Velocity 和二维匀速 Kalman Filter；
- 严格七维历史输入的确定性 GRU；
- train-only mask-aware 归一化、validation early stopping 和安全 checkpoint；
- 单 seed 与三 seed smoke 训练、评价及均值/标准差汇总。

当前尚未完成：

- 概率 GRU、Deep Ensemble 及其他概率预测模型；
- 世界模型；
- 强化学习规划；
- 四旋翼控制；
- ROS 和真机实验。

## 环境要求

正式实验推荐使用 Python 3.10 或 3.11。实验 2 加固先在 Python 3.11.0、NumPy 2.4.6、PyTorch 2.13.0+cpu 环境完成现代兼容验证，最终代码又在 Python 3.10.4、NumPy 2.2.6、PyTorch 2.5.1、pytest 9.1.1 环境完整重跑。Python 3.7 仅保留为历史兼容环境，不再作为正式实验推荐环境。

## 安装方式

从仓库根目录创建独立环境并安装当前实验所需依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 运行方式

生成实验 1 smoke 数据和可视化：

```bash
python scripts/run_experiment_01.py --config configs/experiment_01_smoke.yaml
```

生成并验证实验 1.1 数据集 v2：

```bash
python scripts/generate_dataset_v2.py --config configs/dataset_v2_smoke.yaml
python scripts/validate_dataset_v2.py --dataset-dir outputs/dataset_v2
```

运行实验 2 单 seed smoke：

```bash
python scripts/train_prediction_baseline.py --config configs/experiment_02_smoke.yaml
python scripts/evaluate_prediction_baselines.py --config configs/experiment_02_smoke.yaml
```

运行实验 2 三 seed 管线 smoke：

```bash
python scripts/train_prediction_baseline.py --config configs/experiment_02_multiseed_smoke.yaml
python scripts/evaluate_prediction_baselines.py --config configs/experiment_02_multiseed_smoke.yaml
```

运行全部单元测试：

```bash
python -m pytest -q
```

`tests/conftest.py` 会在测试收集早期把 pytest 进程的 PyTorch intra-op/inter-op 线程固定为 1/1，因此高核机器无需额外环境变量即可稳定运行裸命令；该测试专用限制不修改训练或 formal 配置。以上命令均应从仓库根目录执行。训练随机过程由配置文件中的 `seeds` 序列控制。

## 项目结构

```text
configs/                         实验配置
src/thesis_experiment/data/      轨迹、观测、数据窗口和独立验证
src/thesis_experiment/geometry/  树干生成与遮挡几何
src/thesis_experiment/visualization/  Matplotlib 可视化
src/thesis_experiment/prediction/     经典基线、GRU、归一化和安全训练管线
src/thesis_experiment/evaluation/     预测指标与分层聚合
scripts/                         可直接运行的实验脚本
tests/                           pytest 单元测试
docs/                            验收报告和研究文档
outputs/                         自动生成的实验数据、图片、统计和日志
```

## 实验输出

实验 1 脚本会在 `outputs/experiment_01/` 生成单场景数据。实验 1.1 脚本会在 `outputs/dataset_v2/` 生成 `train.npz`、`validation.npz`、`test.npz`、manifest、日志和 figures。实验 2 脚本会在 `outputs/experiment_02/` 生成各 seed checkpoint、逐层级指标、均值与标准差、运行时间和图表。单 seed 与多 seed smoke 配置共用该目录，后运行的配置会覆盖同名汇总产物。`outputs/` 是自动生成目录，不纳入 Git 版本控制；仓库只保留 `outputs/.gitkeep` 以维持目录结构。

完整的实验 1 验收记录见 [`docs/EXPERIMENT_01_VALIDATION.md`](docs/EXPERIMENT_01_VALIDATION.md)。

实验 1.1 的实测验收记录见 [`docs/DATASET_V2_VALIDATION.md`](docs/DATASET_V2_VALIDATION.md)。

实验 2 的训练前加固与 smoke 验收记录见 [`docs/EXPERIMENT_02_VALIDATION.md`](docs/EXPERIMENT_02_VALIDATION.md)。

## 实验 2 安全约定

所有神经网络输入必须来自 `PredictionDataset` 的五项白名单；`history_true_position`、`trajectory_type`、完整 episode 真值和未来运动参数禁止进入模型。`trajectory_type` 与遮挡分组只在推理完成后用于评价。

PyTorch intra-op 与 inter-op 线程数由 YAML 的 `runtime` 段显式设置并写入日志；两个 smoke 配置实测均为 1/1，formal 配置预设为 4/4 且可按机器调整。模型和 resume checkpoint 只保存 Tensor 与安全的基本容器，NumPy RNG 数组会转换为 Tensor；加载始终显式使用 `weights_only=True`，安全加载失败时不会回退到任意 pickle。过旧且不支持 `weights_only` 的 PyTorch 会被明确拒绝。顶层 `best.pt` 仅供部署便利，正式多 seed 统计必须使用 `checkpoints/seed_<seed>/` 下的全部独立 checkpoint。

## 数据约定

每个样本以历史窗口内最后一个有效带噪观测点作为局部原点。`history_position`、`history_true_position`、`future_position`、`sensor_position` 和 `tree_centers` 使用同一个局部坐标系，`coordinate_origin` 保存恢复世界坐标所需的原点。

缺失历史观测在 `history_position` 中保存为 `NaN`，真实历史和未来监督轨迹仍然保留。`history_occluded` 和 `history_random_dropout` 分别记录树干遮挡与随机丢帧；当前实现只对几何可见帧抽样随机丢帧，因此两种缺失原因互斥。

遮挡采用传感器到目标的闭线段与树干闭圆相交判定。相切以及传感器或目标端点位于圆内均算作相交。

## 数据集 v2 的分层与用途

- scene：独立树干布局和传感器；
- episode：某个 scene 内的一条完整轨迹；
- window：episode 内连续的 20 步历史和 20 步未来。

train、validation 和 test 在生成 scene 前即使用独立 seed 和互斥全局 ID 划分。同一 scene 的所有 episode 与 window 始终位于同一 split，禁止先混合 window 再随机划分。

`history_true_position` 只用于审计和可视化，不能作为后续模型输入；`future_position` 只作为监督标签。

普通预测 window 必须至少有 2 帧可见历史、至少 2 帧连续可见，且每条被接收的 episode 必须产生至少 3 个有效 window。不满足要求的候选 episode 会在有限 `max_attempts` 内重新采样；全程不可见轨迹不会进入普通预测数据集。当前任务定义要求历史窗口至少存在可用观测；全程不可见目标需要其他先验或检测前记忆，不属于本阶段预测问题。

后续模型代码应通过 `thesis_experiment.data.prediction_dataset.PredictionDataset` 读取数据。默认输入严格限定为 `history_position`、`history_velocity`、`history_mask`、派生的 `history_velocity_mask` 和 `time_step_seconds`；NaN 默认填充为 0，并保留相应 mask。标签仅为 `future_position`，索引元数据仅为 `scene_id`、`episode_id` 和 `sample_start_index`。完整 episode 真值、运动参数、`history_true_position` 和 `trajectory_type` 均不得作为默认模型输入。

逐 window 指标可使用 `thesis_experiment.evaluation.aggregate_window_metrics` 聚合。重叠 window 不是独立统计样本，正式论文结果应以 episode 或 scene 等权聚合为默认统计单位。

## 当前限制

- 数据集 v2 仍是二维合成数据，不代表真实林下动力学或传感器；
- smoke 配置为 50 个 scene、每个 scene 3 个 episode，规模只用于管线验证；
- 同一 episode 内 stride=5 的相邻窗口高度重叠，但 scene 级 split 之间无重叠；
- 五种运动是受限运动学模式，尚未覆盖更复杂的交互、目标行为或三维运动；
- 实验 2 的数值仅属于 smoke 工程验证，不是论文正式结论；
- 正式结论必须使用独立 seed 的均值与样本标准差（`ddof=1`），并以 episode 或 scene 为统计单位；单 seed 经典方法的 seed 标准差记为 0，禁止用重叠 window 间的离散程度冒充 seed 不确定性；
- `configs/dataset_v2_formal.yaml` 仅完成加载验证，本轮未生成其 300/50/100 scene 的正式数据。
- `configs/experiment_02_formal.yaml` 已验证可加载，但其依赖的正式 Dataset V2 尚未生成，因此未运行。
