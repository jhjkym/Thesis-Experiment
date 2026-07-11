# Thesis-Experiment

## 当前研究方向

面向复杂林下环境的四旋翼无人机感知、规划与控制研究。

本仓库当前只完成用于后续研究的数据生成基线，不包含轨迹预测、规划训练或无人机控制模型。

## 当前已完成实验

实验 1：林下动态目标遮挡场景与观测数据生成验证。

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

当前尚未完成：

- 多场景、多运动模式数据集；
- 卡尔曼滤波预测；
- GRU 及概率预测模型；
- 世界模型；
- 强化学习规划；
- 四旋翼控制；
- ROS 和真机实验。

## 环境要求

推荐使用 Python 3.10 或 3.11。当前基线也已在 Python 3.7.0、NumPy 1.21.6、Matplotlib 3.5.3、PyYAML 6.0.1 和 pytest 7.4.4 环境完成实际验证。

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

运行全部单元测试：

```bash
python -m pytest -q
```

以上命令均应从仓库根目录执行。所有随机过程由配置文件中的 `seed` 控制。

## 项目结构

```text
configs/                         实验配置
src/thesis_experiment/data/      轨迹、观测和数据窗口
src/thesis_experiment/geometry/  树干生成与遮挡几何
src/thesis_experiment/visualization/  Matplotlib 可视化
scripts/                         可直接运行的实验脚本
tests/                           pytest 单元测试
docs/                            验收报告和研究文档
outputs/                         自动生成的实验数据、图片、统计和日志
```

## 实验输出

运行脚本会在 `outputs/experiment_01/` 生成 `dataset.npz`、三张示例图、统计 JSON 和运行日志。`outputs/` 是自动生成目录，不纳入 Git 版本控制；仓库只保留 `outputs/.gitkeep` 以维持目录结构。

完整的实验 1 验收记录见 [`docs/EXPERIMENT_01_VALIDATION.md`](docs/EXPERIMENT_01_VALIDATION.md)。

## 数据约定

每个样本以历史窗口内最后一个有效带噪观测点作为局部原点。`history_position`、`history_true_position`、`future_position`、`sensor_position` 和 `tree_centers` 使用同一个局部坐标系，`coordinate_origin` 保存恢复世界坐标所需的原点。

缺失历史观测在 `history_position` 中保存为 `NaN`，真实历史和未来监督轨迹仍然保留。`history_occluded` 和 `history_random_dropout` 分别记录树干遮挡与随机丢帧；当前实现只对几何可见帧抽样随机丢帧，因此两种缺失原因互斥。

遮挡采用传感器到目标的闭线段与树干闭圆相交判定。相切以及传感器或目标端点位于圆内均算作相交。

## 当前限制

- 当前 100 个样本来自单场景、单条匀速轨迹；
- 样本由相邻滑动窗口产生，窗口之间高度重叠；
- 当前数据不能直接作为无信息泄漏的训练集、验证集和测试集；
- 下一步需要先按 scene 和 episode 构建并划分多场景数据集。
