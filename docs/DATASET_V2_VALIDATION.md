# 数据集 v2 训练前加固验收报告

- 对象：实验 1.1——多场景、多运动模式、无数据泄漏的数据集构建
- 日期：2026-07-12（Asia/Shanghai）
- smoke 配置：`configs/dataset_v2_smoke.yaml`
- Python：`/home/tom/.venv/bin/python`，Python 3.7.0
- 分支：`feature/dataset-v2`
- 结论：**训练前数据边界与审计项通过；具备进入轨迹预测基线实验的数据管线条件。本轮未实现预测模型。**

## 1. 实际命令与退出状态

```bash
python scripts/generate_dataset_v2.py --config configs/dataset_v2_smoke.yaml
```

退出状态 `0`。生成 50 个 scene、150 个接收 episode、1,260 个有效 window 和 9 张图片。

```bash
python scripts/validate_dataset_v2.py --dataset-dir outputs/dataset_v2
```

退出状态 `0`，结果 `status: passed`。

```bash
python -m pytest -q
```

退出状态 `0`：

```text
108 passed in 2.82s
```

失败 0，跳过 0。

`configs/dataset_v2_formal.yaml` 已实际加载并通过配置测试，但按要求未生成正式规模数据。

## 2. 数据规模、隔离与 seed

| split | seed | scene | episode | 有效 window |
|---|---:|---:|---:|---:|
| train | 20261101 | 30 | 90 | 753 |
| validation | 20261201 | 10 | 30 | 249 |
| test | 20261301 | 10 | 30 | 258 |

三个 split 使用不同根 seed，并在 scene 生成前完成划分。检查结果：

- scene ID 两两交集均为空；
- episode ID 两两交集均为空；
- 传感器位置、树心和树半径组成的场景内容指纹两两交集均为 0；
- 同一 episode 不跨 split，window 不跨 episode；
- 每个接收 episode 至少产生 3 个有效 window；
- 每个 split 在 episode 和有效 window 两个层级均包含五种运动。

固定 seed 的自动化测试逐字段使用 `equal_nan=True` 验证重复生成一致；修改 split seed 后三个 split 的随机场景、轨迹和观测均发生变化。传感器净距的固定 seed 复现测试也通过。

## 3. 历史有效性与拒绝重采样

配置门槛：

```yaml
window:
  history_steps: 20
  future_steps: 20
  window_stride: 5
  minimum_visible_history_steps: 2
  minimum_consecutive_visible_steps: 2
  minimum_windows_per_episode: 3
```

每个候选 window 同时满足总可见数和最长连续可见数门槛。候选 episode 若不足 3 个有效 window，会在 `trajectory.max_attempts=1000` 内重新采样；达到上限会抛出带运动类型、尝试次数和最后原因的 `RuntimeError`。

| split | 全程不可见候选 | 历史不足候选 | 物理约束失败候选 |
|---|---:|---:|---:|
| train | 11 | 4 | 62 |
| validation | 3 | 2 | 9 |
| test | 4 | 1 | 22 |

前两列是“观测有效性拒绝 episode 候选”，第三列是轨迹越界、碰撞或运动上限等物理拒绝候选，manifest 分开保存，未混为一类。所有最终接收 episode 均可观测并满足最少 window 数。

当前任务定义要求历史窗口至少存在可用观测。全程不可见目标需要其他先验或检测前记忆，不属于本阶段普通轨迹预测问题，因此不会作为普通预测样本保存。

## 4. 运动类型、有效 window 与拒绝统计

每类 episode 数量严格均衡：train 各 18，validation 各 6，test 各 6。

表中“观测拒绝”是全程不可见与历史不足之和；“物理拒绝”单列。

| split | 类型 | episode | 有效 window | 观测拒绝 | 物理拒绝 |
|---|---|---:|---:|---:|---:|
| train | constant_velocity | 18 | 162 | 4 | 18 |
| train | constant_acceleration | 18 | 147 | 3 | 17 |
| train | constant_turn | 18 | 153 | 1 | 7 |
| train | stop_and_go | 18 | 150 | 6 | 15 |
| train | piecewise_direction | 18 | 141 | 1 | 5 |
| validation | constant_velocity | 6 | 53 | 2 | 2 |
| validation | constant_acceleration | 6 | 53 | 1 | 1 |
| validation | constant_turn | 6 | 39 | 0 | 1 |
| validation | stop_and_go | 6 | 53 | 0 | 3 |
| validation | piecewise_direction | 6 | 51 | 2 | 2 |
| test | constant_velocity | 6 | 54 | 0 | 2 |
| test | constant_acceleration | 6 | 53 | 1 | 3 |
| test | constant_turn | 6 | 48 | 4 | 6 |
| test | stop_and_go | 6 | 54 | 0 | 6 |
| test | piecewise_direction | 6 | 49 | 0 | 5 |

按每个 window 历史段内“最长连续几何遮挡”分箱，顺序为 `0 / 1-5 / 6-10 / 11-15 / 16-20`：

| split | 类型 | 分箱计数 |
|---|---|---|
| train | constant_velocity | 148 / 5 / 4 / 4 / 1 |
| train | constant_acceleration | 126 / 6 / 7 / 6 / 2 |
| train | constant_turn | 143 / 3 / 3 / 3 / 1 |
| train | stop_and_go | 139 / 3 / 3 / 3 / 2 |
| train | piecewise_direction | 107 / 9 / 10 / 10 / 5 |
| validation | constant_velocity | 49 / 1 / 1 / 1 / 1 |
| validation | constant_acceleration | 50 / 1 / 1 / 1 / 0 |
| validation | constant_turn | 22 / 4 / 5 / 4 / 4 |
| validation | stop_and_go | 47 / 2 / 2 / 2 / 0 |
| validation | piecewise_direction | 48 / 1 / 1 / 1 / 0 |
| test | constant_velocity | 54 / 0 / 0 / 0 / 0 |
| test | constant_acceleration | 48 / 2 / 2 / 1 / 0 |
| test | constant_turn | 36 / 3 / 3 / 3 / 3 |
| test | stop_and_go | 51 / 3 / 0 / 0 / 0 |
| test | piecewise_direction | 46 / 1 / 1 / 1 / 0 |

没有任何运动类型因可见性过滤而从 split 中消失。

## 5. 历史可见性审计字段

新增的 window 级字段：

- `history_visible_count`：历史 20 步可见帧数；
- `last_valid_observation_age_steps`：最后有效观测距 history 末尾的步数；
- `valid_velocity_count`：由相邻两帧有效历史观测计算出的速度数量；
- `history_max_consecutive_occlusion_steps`：最长连续几何遮挡；
- `occlusion_length_bin`：上述长度的区间编码。

独立重算得到的分布如下。

`history_visible_count`：

```text
train:      2:4, 3:5, 4:3, 5:8, 6:2, 7:7, 8:5, 9:5, 10:9, 11:6,
            12:6, 13:2, 14:6, 15:10, 16:9, 17:41, 18:114, 19:248, 20:263
validation: 2:1, 3:1, 4:3, 5:2, 6:4, 8:3, 10:2, 11:4, 13:4, 14:3,
            15:3, 16:4, 17:17, 18:62, 19:80, 20:56
test:       2:1, 4:2, 5:1, 6:2, 9:2, 10:3, 11:1, 14:3, 15:3, 16:3,
            17:13, 18:52, 19:81, 20:91
```

`last_valid_observation_age_steps`：

```text
train:      0:682, 1:29, 2:8, 3:3, 4:1, 5:2, 6:2, 7:4, 8:3, 9:1,
            10:2, 11:2, 12:4, 13:2, 14:1, 15:2, 16:1, 17:3, 18:1
validation: 0:206, 1:20, 2:2, 3:1, 4:1, 5:1, 6:3, 7:2, 8:1, 9:1,
            10:1, 11:2, 12:2, 13:1, 14:1, 15:1, 16:2, 17:1
test:       0:229, 1:16, 3:1, 4:1, 5:2, 6:1, 8:1, 9:1, 10:1, 11:1,
            13:1, 15:1, 16:1, 18:1
```

无有效速度 window 数量：train `0`、validation `0`、test `0`。这是 `minimum_consecutive_visible_steps=2` 的直接结果；完整的有效速度数量分布保存在 manifest。

## 6. 严格预测数据加载边界

`src/thesis_experiment/data/prediction_dataset.py` 的默认输出分为三部分。

允许的模型输入仅为：

```text
history_position
history_velocity
history_mask
history_velocity_mask
time_step_seconds
```

监督标签仅为 `future_position`。索引元数据仅为 `scene_id`、`episode_id`、`sample_start_index`。

`history_velocity_mask` 不从 NPZ 读取，而是由 `history_velocity` 两个分量均有限计算。`history_position` 和 `history_velocity` 中的非有限值默认填充为 0，也可配置其他有限填充值；原始 `history_mask` 与派生速度 mask 均保留。加载器还拒绝非二值 mask、位置/mask 不一致、非正时间步和非有限标签。

禁止作为模型输入的字段包括但不限于：

```text
history_true_position
trajectory_type
episode_true_position_world
episode_acceleration_world
episode_velocity_world
episode_turn_rate
episode_stop_start_time
episode_stop_duration
episode_piecewise_turn_time
episode_piecewise_turn_angle
episode_acceleration_parameter
```

白名单是封闭的：即使未知字段未出现在显式禁止集合中，只要不在五项输入白名单中也会被拒绝。单元测试验证默认样本不返回任何禁止字段。

## 7. 字段、shape 与 dtype

三个 NPZ 各含 46 个保存字段。记 `N=753/249/258`、`E=90/30/30`、`S=30/10/10`（train/validation/test），`H=20`、`F=20`、`T=80`、`K=16`。

| 字段组 | shape | dtype |
|---|---|---|
| `history_position`, `history_velocity`, `history_true_position` | `(N,H,2)` | float64 |
| `history_mask`, `history_occluded`, `history_random_dropout` | `(N,H)` | uint8 |
| `future_position` | `(N,F,2)` | float64 |
| `coordinate_origin`, `sensor_position` | `(N,2)` | float64 |
| `tree_centers` / `tree_radii` | `(N,K,2)` / `(N,K)` | float64 |
| `scene_id`, `episode_id`, `sample_start_index` | `(N,)` | int64 |
| `trajectory_type`, `occlusion_length_bin` | `(N,)` | int8 |
| `history_visible_count`, `last_valid_observation_age_steps`, `valid_velocity_count`, `history_max_consecutive_occlusion_steps` | `(N,)` | int16 |
| `history_start_time`, `future_start_time`, `time_step_seconds` | `(N,)` | float64 |
| `scene_ids` | `(S,)` | int64 |
| `scene_sensor_position_world` | `(S,2)` | float64 |
| `scene_tree_centers_world` / `scene_tree_radii` | `(S,K,2)` / `(S,K)` | float64 |
| `episode_ids`, `episode_scene_ids` | `(E,)` | int64 |
| `episode_trajectory_types` | `(E,)` | int8 |
| `episode_true_position_world`, `episode_observed_position_world`, `episode_velocity_world`, `episode_acceleration_world` | `(E,T,2)` | float64 |
| `episode_visible_mask`, `episode_occluded_mask`, `episode_random_dropout_mask` | `(E,T)` | uint8 |
| `episode_times` | `(E,T)` | float64 |
| `episode_initial_position`, `episode_initial_velocity`, `episode_acceleration_parameter` | `(E,2)` | float64 |
| `episode_turn_rate`, `episode_stop_start_time`, `episode_stop_duration`, `episode_piecewise_turn_time`, `episode_piecewise_turn_angle` | `(E,)` | float64 |

派生的 `history_velocity_mask` 是加载器输出，不重复保存到 NPZ。

## 8. 传感器净距、轨迹和遮挡几何

配置要求 `sensor_tree_clearance=0.5 m`。独立复算的传感器到最近树干表面距离：

| split | 最小传感器树表净距 | 最小轨迹树表净距 | 最大速度 | 最大加速度 |
|---|---:|---:|---:|---:|
| train | 0.5464014664 m | 0.2022624073 m | 0.9388930429 m/s | 0.9916200186 m/s² |
| validation | 0.9956997870 m | 0.2433272706 m | 0.7188590657 m/s | 1.0370124732 m/s² |
| test | 0.5308041844 m | 0.3811195437 m | 1.1288849519 m/s | 1.0394245112 m/s² |

要求上限为 1.20 m/s 和 1.50 m/s²。所有 episode 均有限、不越界、不穿树、满足 0.20 m 轨迹树表安全距离；相邻采样点线段也对扩张圆检查。违规 episode 数为 0。

验证器按保存的传感器、树木和完整真值重新计算 12,000 帧遮挡：train 7,200、validation 2,400、test 2,400，标记差异总数为 0。窗口局部树木、传感器、观测与对应 world 数组的最大重建误差不超过 `3.552713678800501e-15`。

## 9. 窗口、未来信息与观测一致性

- history 20 步、future 20 步、10 Hz、stride 5；
- future 第一帧紧接 history 最后一帧，不重叠一帧；
- 所有切窗严格位于一个 episode 内；
- `history_velocity` 只用历史内相邻有效观测后向差分；
- 局部原点只用 history 最后有效观测，不使用 future；
- 遮挡历史未用真值或未来插值；
- `history_true_position` 仅审计，`future_position` 仅监督；
- 停止时间、转弯时间、加速度参数等未来运动参数不在输入白名单；
- `trajectory_type` 仅元数据，不是默认输入。

世界真值最大重建误差 `1.1102230246251565e-16`，历史速度最大复算误差 `2.220446049250313e-15`。

相邻有效 window 对均因 stride 5 而重叠：train 663/663、validation 219/219、test 228/228。重叠只在同一 episode 内。

## 10. 观测比例与 manifest 一致性

| split | 可见比例 | 几何遮挡比例 | 随机丢帧比例 |
|---|---:|---:|---:|
| train | 0.9013280212 | 0.0549800797 | 0.0436918991 |
| validation | 0.8809236948 | 0.0598393574 | 0.0592369478 |
| test | 0.9180232558 | 0.0368217054 | 0.0451550388 |

几何遮挡和随机丢帧原因互斥，二者与 `history_mask` 完全一致；`history_mask=0` 时位置为 NaN，`history_mask=1` 时位置有限；完整真值和 future 始终有限。

验证器从 NPZ 独立重算并与 manifest 比较：scene/episode/window 数量、运动类型、三种观测比例、速度/加速度范围、完整 episode 连续遮挡长度、window 遮挡分箱、三个历史审计分布、拒绝统计、窗口重叠、全部字段 shape/dtype。差异为 0。

## 11. 聚合评价策略

`src/thesis_experiment/evaluation/aggregation.py` 提供 `aggregate_window_metrics`，输入未来逐 window 指标后返回：

- window 直接均值；
- episode 内先平均、再对 episode 等权的均值；
- scene 内先平均、再对 scene 等权的均值；
- 按 `trajectory_type` 分组；
- 按遮挡长度区间分组。

函数验证 episode 只能映射到一个 scene 和一种运动类型。window 均值及分组 window 均值是描述性统计；正式论文默认以 episode 或 scene 等权均值为统计单位，不把重叠 window 当作独立样本。

## 12. 保存后可视化

可视化入口只读取已保存 NPZ 和 manifest，不导入场景或轨迹生成器，也不调用随机数。实际示例：

| 类型 | split | scene | episode | window 索引 / 起点 | 可见 | 遮挡 | 丢帧 |
|---|---|---:|---:|---|---:|---:|---:|
| constant_velocity | train | 17 | 53 | 443 / 0 | 42 | 37 | 1 |
| constant_acceleration | validation | 33 | 99 | 71 / 0 | 39 | 39 | 2 |
| constant_turn | test | 47 | 141 | 178 / 0 | 34 | 43 | 3 |
| stop_and_go | train | 19 | 57 | 476 / 10 | 52 | 26 | 2 |
| piecewise_direction | test | 40 | 122 | 18 / 25 | 37 | 40 | 3 |

九张图包含五类轨迹示例、运动类型分布、遮挡长度分布、速度分布和 split 对比；树干使用等比例坐标和真实半径。

## 13. 输出文件

| 文件 | bytes |
|---|---:|
| `train.npz` | 1,203,191 |
| `validation.npz` | 396,688 |
| `test.npz` | 416,894 |
| `dataset_manifest.json` | 47,331 |
| `run.log` | 2,019 |
| `figures/trajectory_constant_velocity.png` | 106,098 |
| `figures/trajectory_constant_acceleration.png` | 105,960 |
| `figures/trajectory_constant_turn.png` | 105,138 |
| `figures/trajectory_stop_and_go.png` | 109,106 |
| `figures/trajectory_piecewise_direction.png` | 107,595 |
| `figures/trajectory_type_distribution.png` | 69,002 |
| `figures/occlusion_length_distribution.png` | 51,007 |
| `figures/speed_distribution.png` | 41,875 |
| `figures/split_statistics_comparison.png` | 84,689 |

## 14. 发现并修复的问题

本轮修复并增加回归测试：

1. 加入严格模型输入/标签/索引白名单，禁止完整 episode 真值和未来运动参数泄漏；
2. 对位置和速度缺失值做有限填充并保留 mask；
3. 增加历史有效性双门槛、每 episode 最少有效 window 和有限次数重采样；
4. 全程不可见与历史不足 episode 候选改为拒绝并分开统计；
5. 树木生成加入传感器到树表面的 0.5 m 净距；
6. 增加可见数、末次观测年龄、有效速度数和遮挡区间审计字段；
7. 增加 episode/scene 等权评价聚合；
8. 配置拒绝 NaN/Inf，使用实际采样时间数量校验 window 可行性，并提前保证每 split 至少五个 episode；
9. 验证器新增 mask 二值性、场景内容重复、局部场景恢复和全帧遮挡几何重算；
10. 明确区分观测有效性拒绝与物理候选拒绝，避免统计口径含混。

## 15. 尚未解决的问题与阶段结论

尚有限制：

- 数据仍为二维合成运动学和圆形树干；
- stride 5 使同 episode window 高度重叠，不能作为独立样本解释；
- smoke 只用于管线验证；`dataset_v2_formal.yaml` 虽已验证可加载，但尚未实际生成正式规模数据；
- 全程不可见追踪需要检测前记忆或其他先验，不属于当前普通预测任务；
- 按运动类型和遮挡区间的直接 window 均值仅用于描述，正式显著性分析仍应以 episode 或 scene 为统计单位。

**阶段结论：可以进入轨迹预测基线实验，但必须强制使用严格加载白名单，并以 episode 或 scene 为默认评价单位。** 本轮到此停止，未实现卡尔曼滤波、GRU、PyTorch、强化学习、世界模型、ROS 或控制器。
