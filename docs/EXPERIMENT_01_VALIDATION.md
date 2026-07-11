# 实验 1 完整验收报告

- 验收对象：林下动态目标遮挡数据生成与可视化
- 验收日期：2026-07-11（Asia/Shanghai）
- 项目目录：`/home/tom/Thesis-Experiment`
- 最终结论：**通过（修复一项可视化可追溯性问题后）**
- 范围声明：本次只验收并修复实验 1；未实现或启动卡尔曼滤波、GRU、强化学习、世界模型、ROS 或控制实验。

## 1. 实际环境

以下命令均在项目根目录实际执行，退出状态均为 `0`：

```bash
pwd
python --version
which python
pip freeze
git status --short
git log -1 --oneline
find . -maxdepth 3 -type f | sort
```

结果：

```text
pwd:            /home/tom/Thesis-Experiment
python:         Python 3.7.0
which python:   /home/tom/.venv/bin/python
git log -1:     211f052 first commit
```

项目当前改动尚未提交。最终 `git status --short`：

```text
 M README.md
?? .gitignore
?? configs/
?? docs/
?? outputs/
?? requirements.txt
?? scripts/
?? src/
?? tests/
```

`pip freeze` 退出状态为 `0`；pip 同时提示 `/home/tom/.cache/pip` 不可写并禁用缓存，该提示不影响依赖读取或实验运行。完整输出：

```text
actionlib==1.14.3
angles==1.9.14
bondpy==1.8.7
camera-calibration==1.17.0
camera-calibration-parsers==1.12.1
catkin==0.8.12
controller-manager==0.20.0
controller-manager-msgs==0.20.0
cv-bridge==1.16.2
cycler==0.11.0
diagnostic-analysis==1.12.1
diagnostic-common-diagnostics==1.12.1
diagnostic-updater==1.12.1
dynamic-reconfigure==1.7.6
exceptiongroup==1.3.1
fonttools==4.38.0
gazebo_plugins==2.9.3
gazebo_ros==2.9.3
gencpp==0.7.2
geneus==3.0.0
genlisp==0.4.18
genmsg==0.6.1
gennodejs==2.0.2
genpy==0.6.18
image-geometry==1.16.2
importlib-metadata==6.7.0
iniconfig==2.0.0
interactive-markers==1.12.2
joint-state-publisher==1.15.2
joint-state-publisher-gui==1.15.2
kiwisolver==1.4.5
laser-geometry==1.6.8
matplotlib==3.5.3
mavros==1.20.1
message-filters==1.17.4
numpy==1.21.6
packaging==24.0
Pillow==9.5.0
pluggy==1.2.0
pyparsing==3.1.4
pytest==7.4.4
python-dateutil==2.9.0.post0
python-qt-binding==0.4.6
PyYAML==6.0.1
qt-dotgraph==0.4.5
qt-gui==0.4.5
qt-gui-cpp==0.4.5
qt-gui-py-common==0.4.5
resource-retriever==1.12.10
rosbag==1.17.4
rosboost-cfg==1.15.10
rosclean==1.15.10
roscreate==1.15.10
rosgraph==1.17.4
roslaunch==1.17.4
roslib==1.15.10
roslint==0.12.0
roslz4==1.17.4
rosmake==1.15.10
rosmaster==1.17.4
rosmsg==1.17.4
rosnode==1.17.4
rosparam==1.17.4
rospy==1.17.4
rosservice==1.17.4
rostest==1.17.4
rostopic==1.17.4
rosunit==1.15.10
roswtf==1.17.4
rqt-action==0.4.11
rqt-bag==0.5.3
rqt-bag-plugins==0.5.3
rqt-console==0.4.14
rqt-dep==0.4.14
rqt-graph==0.4.16
rqt-gui==0.5.5
rqt-image-view==0.4.19
rqt-launch==0.4.10
rqt-logger-level==0.4.13
rqt-moveit==0.5.13
rqt-msg==0.4.12
rqt-nav-view==0.5.8
rqt-plot==0.4.16
rqt-pose-view==0.5.13
rqt-publisher==0.4.12
rqt-py-common==0.5.5
rqt-py-console==0.4.12
rqt-reconfigure==0.5.7
rqt-robot-dashboard==0.5.8
rqt-robot-monitor==0.5.15
rqt-robot-steering==0.5.14
rqt-runtime-monitor==0.5.10
rqt-rviz==0.7.2
rqt-service-caller==0.4.12
rqt-shell==0.4.13
rqt-srv==0.4.11
rqt-tf-tree==0.6.5
rqt-top==0.4.11
rqt-topic==0.4.15
rqt-web==0.4.11
rviz==1.14.26
sensor-msgs==1.13.2
six==1.17.0
smach==2.5.3
smach-ros==2.5.3
smclib==1.8.7
tf==1.13.4
tf-conversions==1.13.4
tf2-geometry-msgs==0.7.10
tf2-kdl==0.7.10
tf2-py==0.7.10
tf2-ros==0.7.10
tomli==2.0.1
topic-tools==1.17.4
typing_extensions==4.7.1
xacro==1.14.20
zipp==3.15.0
```

最终执行 `find . -maxdepth 3 -type f | sort` 的结果：

```text
./.git/HEAD
./.git/config
./.git/description
./.git/gk/config
./.git/hooks/applypatch-msg.sample
./.git/hooks/commit-msg.sample
./.git/hooks/fsmonitor-watchman.sample
./.git/hooks/post-update.sample
./.git/hooks/pre-applypatch.sample
./.git/hooks/pre-commit.sample
./.git/hooks/pre-merge-commit.sample
./.git/hooks/pre-push.sample
./.git/hooks/pre-rebase.sample
./.git/hooks/pre-receive.sample
./.git/hooks/prepare-commit-msg.sample
./.git/hooks/update.sample
./.git/index
./.git/info/exclude
./.git/logs/HEAD
./.git/packed-refs
./.gitignore
./.pytest_cache/.gitignore
./.pytest_cache/CACHEDIR.TAG
./.pytest_cache/README.md
./README.md
./configs/experiment_01_smoke.yaml
./docs/EXPERIMENT_01_VALIDATION.md
./outputs/experiment_01/dataset.npz
./outputs/experiment_01/mask_timeline.png
./outputs/experiment_01/run.log
./outputs/experiment_01/sample_statistics.json
./outputs/experiment_01/scene_example.png
./outputs/experiment_01/trajectory_example.png
./requirements.txt
./scripts/__pycache__/run_experiment_01.cpython-37.pyc
./scripts/run_experiment_01.py
./src/thesis_experiment/__init__.py
./src/thesis_experiment/config.py
./tests/__pycache__/conftest.cpython-37-pytest-7.4.4.pyc
./tests/__pycache__/conftest.cpython-37.pyc
./tests/__pycache__/test_experiment_01.cpython-37-pytest-7.4.4.pyc
./tests/__pycache__/test_experiment_01.cpython-37.pyc
./tests/conftest.py
./tests/test_experiment_01.py
```

该命令按要求限制为 `maxdepth 3`，因此不会列出位于更深层级的 `data/`、`geometry/` 和 `visualization/` 模块文件。

## 2. 正式实验与测试命令

修复后最终实际执行：

| 命令 | 退出状态 | 实际结果 |
|---|---:|---|
| `python scripts/run_experiment_01.py --config configs/experiment_01_smoke.yaml` | 0 | 生成 25 棵树、201 个轨迹时刻、100 个窗口及全部输出；日志结束于 `Experiment 01 completed successfully` |
| `python -m pytest -q` | 0 | `15 passed in 0.11s` |

pytest 汇总：

```text
passed:  15
failed:  0
skipped: 0
```

验收初次运行时为 `14 passed`；发现并修复可视化样本不可追溯问题后新增 1 项回归测试，最终为 15 项通过。

## 3. 实验输出

实际执行：

```bash
ls -lh outputs/experiment_01/
```

退出状态：`0`。输出：

```text
total 352K
-rw-r--r-- 1 tom tom  89K Jul 11 23:19 dataset.npz
-rw-r--r-- 1 tom tom  37K Jul 11 23:19 mask_timeline.png
-rw-r--r-- 1 tom tom 1.2K Jul 11 23:19 run.log
-rw-r--r-- 1 tom tom 1.3K Jul 11 23:19 sample_statistics.json
-rw-r--r-- 1 tom tom 101K Jul 11 23:19 scene_example.png
-rw-r--r-- 1 tom tom 106K Jul 11 23:19 trajectory_example.png
```

精确文件大小：

| 文件 | 字节数 |
|---|---:|
| `dataset.npz` | 90,210 |
| `scene_example.png` | 103,313 |
| `trajectory_example.png` | 107,858 |
| `mask_timeline.png` | 37,675 |
| `sample_statistics.json` | 1,287 |
| `run.log` | 1,217 |

六个文件均真实存在、非空，且修改时间来自修复后的同一次 23:19 正式运行。

## 4. 数据集内容验收

使用 `numpy.load(path, allow_pickle=False)` 实际读取 `dataset.npz`，退出状态为 `0`。

### 4.1 字段、shape 和 dtype

| 字段 | shape | dtype |
|---|---|---|
| `history_position` | `(100, 20, 2)` | `float64` |
| `history_true_position` | `(100, 20, 2)` | `float64` |
| `history_velocity` | `(100, 20, 2)` | `float64` |
| `history_mask` | `(100, 20)` | `uint8` |
| `history_occluded` | `(100, 20)` | `uint8` |
| `history_random_dropout` | `(100, 20)` | `uint8` |
| `future_position` | `(100, 20, 2)` | `float64` |
| `sensor_position` | `(100, 2)` | `float64` |
| `tree_centers` | `(100, 25, 2)` | `float64` |
| `tree_radii` | `(100, 25)` | `float64` |
| `coordinate_origin` | `(100, 2)` | `float64` |
| `scene_id` | `(100,)` | `int64` |
| `sample_start_index` | `(100,)` | `int64` |
| `time_step_seconds` | `()` | `float64` |

字段总数：`14`。

### 4.2 NaN、mask、监督标签和场景

```text
history_position 标量总数: 4000
history_position NaN 数量: 1034
history_mask=0: 517 个时间槽
history_mask=1: 1483 个时间槽

mask=0 -> 两个 position 坐标均为 NaN: True
mask=1 -> 两个 position 坐标均为有限值: True
mask=0 -> history_true_position 仍为有限真值: True

future_position NaN 数量: 0
future_position Inf 数量: 0
future_position 全部有限: True

scene_id 唯一值: [0]
scene_id 唯一值数量: 1
```

因此 `1034 = 517 * 2`，历史缺失槽与二维观测 NaN 严格对应；未来监督轨迹没有 NaN 或 Inf。

### 4.3 树数组范围

数据集保存局部坐标，因此同时报告局部值和加回 `coordinate_origin` 后的世界值：

```text
tree_centers 局部 x: [-10.144207292881239, 13.962437898895304]
tree_centers 局部 y: [ -5.672190414853663, 11.190772629063302]

tree_centers 世界 x: [0.8206299527853957, 18.820412363494608]
tree_centers 世界 y: [0.6794451998870175, 14.160495965271249]

tree_radii: [0.3145848776586147, 0.6423006520473906]
```

### 4.4 随机抽取 3 个样本

使用固定验收 RNG：

```python
np.random.default_rng(20260711).choice(100, 3, replace=False)
# array([19, 82, 17])
```

实际 mask：

```text
sample[19] = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
sample[82] = [1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1]
sample[17] = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
```

### 4.5 世界坐标重建

独立按配置的 `p(t)=p0+vt` 生成 201 步原始轨迹，并按每个 `sample_start_index` 取得期望历史/未来片段；再将局部数据加回每个样本的 `coordinate_origin`：

```text
history 最大绝对重建误差: 3.552713678800501e-15
future  最大绝对重建误差: 3.552713678800501e-15
综合最大绝对重建误差:    3.552713678800501e-15
atol=1e-12 下 history 匹配: True
atol=1e-12 下 future 匹配:  True
```

这同时验证了 `future_position` 确实保存真实监督轨迹，而不是观测轨迹。

## 5. 可复现性验收

使用同一个运行脚本连续运行三次。前两次所有仿真配置和 seed 相同，仅临时输出路径不同；第三次仅将 seed 从 `20260712` 改为 `20260713`。三个进程退出状态均为 `0`。

临时目录：

```text
/tmp/experiment_01_final_repro_erjykuv1/same_seed_a
/tmp/experiment_01_final_repro_erjykuv1/same_seed_b
/tmp/experiment_01_final_repro_erjykuv1/changed_seed
```

比较使用 `np.testing.assert_allclose(rtol=0, atol=0, equal_nan=True)`，因此 NaN 按相等处理且有限值要求逐位完全相同：

| 字段 | 同 seed 两次一致 | 改 seed 后仍一致 |
|---|---:|---:|
| `history_position` | True | False |
| `history_velocity` | True | False |
| `history_mask` | True | False |
| `future_position` | True | False |
| `tree_centers` | True | False |
| `tree_radii` | True | False |

改 seed 后六个字段不再全部一致：`False`。可复现性及 seed 生效检查通过。

## 6. 遮挡几何逻辑

实际导入 `segment_intersects_circle`、`segment_intersects_circles` 和 `is_occluded` 运行以下案例，验收脚本退出状态为 `0`：

| 案例 | 实际结果 | 期望 | 结论 |
|---|---:|---:|---|
| 线段与圆不相交 | False | False | PASS |
| 线段穿过圆心 | True | True | PASS |
| 线段与圆相切 | True | True | PASS |
| 圆只位于线段延长线上 | False | False | PASS |
| 传感器端点位于圆内 | True | True | PASS |
| 目标端点位于圆内 | True | True | PASS |
| 多树逐树结果 | `[True, False, True, False]` | 同左 | PASS |
| 多树总体遮挡（`any`） | True | True | PASS |

当前定义使用“闭线段与闭圆相交”：最近距离满足 `distance <= radius + atol` 即相交。因此：

- **相切算作遮挡**；
- 传感器或目标在圆内均算遮挡；
- 投影截断到 `[0,1]`，只有延长线相交但不接触有限线段时不遮挡；
- 多树时任一树相交即遮挡。

树生成器本身禁止树干包含或接触传感器，所以“传感器在圆内=True”是通用几何函数面对非法或外部输入时的明确语义。README、函数 docstring、`<=` 实现和相切单元测试现已一致。

## 7. 树干生成约束

按 smoke 配置 seed `20260712` 独立重建森林，并将数据集局部树坐标恢复到世界坐标交叉验证：

```text
实际树数: 25
配置树数: 25
实际半径范围: [0.314584877659, 0.642300652047]
配置半径范围: [0.3, 0.65]

所有树完整位于边界内: True
最小边界余量: 0.143713053958

没有树覆盖或接触传感器: True
传感器到最近树干表面的距离: 0.827228952278

任意树间净距离满足配置: True
最小树间净距离: 0.475920403080
配置最小净距离: 0.300000000000

数据集树世界坐标最大误差: 1.77635683940025e-15
数据集半径最大误差: 0
数据集传感器世界坐标最大误差: 0
```

过严约束的有限失败测试：

```text
scene_size=(2.0, 2.0)
tree_count=4
radius_range=(0.49, 0.49)
min_spacing=1.0
max_attempts=7

exception_type=RuntimeError
exception_message=could place only 1 of 4 trunks after 7 attempts; reduce tree_count, radii, or min_spacing, or increase max_attempts
elapsed_seconds=0.000243271
```

生成器受 `max_attempts` 限制，会明确抛出 `RuntimeError`，不会无限循环。

## 8. 数据窗口验收

```text
轨迹总步数 T: 201
duration * sample_rate + 1: 201
history_steps: 20
future_steps: 20

原始连续窗口数 T-H-F+1: 162
历史内至少有一个有效观测、可定义局部原点的候选窗口: 120
配置请求样本数: 100
实际保存样本数: 100

sample_start_index 最小值: 0
sample_start_index 最大值: 155
sample_start_index 唯一值数: 100
保存的 starts 与确定性均匀选择算法一致: True
```

逐样本验证结果：

- 历史内部时间索引连续：True；
- 未来内部时间索引连续：True；
- 未来首帧严格等于历史末帧下一帧，无错位也无重叠：True；
- 恢复世界坐标后的历史/未来均与对应原始轨迹片段一致：True；
- `history_velocity` 仅等于历史窗口内相邻可见观测的后向差分：True；
- 每个窗口首帧速度为 NaN，未读取窗口前数据或未来轨迹：True；
- `scene_id` 唯一值为 `[0]`，所有窗口来自同一 scene，未发生跨 scene 拼接。

样本 0 的原始索引：

```text
sample_start_index: 0
history index range: [0, 19]
future index range:  [20, 39]
history time range:  [0.0, 1.9] s
future time range:   [2.0, 3.9] s
```

当前 smoke 配置只生成一个 scene，因此“跨不同 scene 边界”的错误在该产物中不存在，但也无法用此单场景产物触发多场景边界案例。这不影响本轮单场景需求。

## 9. 统计文件一致性

从 NPZ 独立计算，而非调用生成统计的函数：

```text
总历史时间槽: 2000
mask=1 数量: 1483
mask=0 数量: 517
mask=1 比例: 0.7415
mask=0 比例: 0.2585
history_position NaN 比例: 0.2585

几何遮挡数量: 468 (比例 0.234)
随机丢帧数量: 49 (比例 0.0245)
两者同时发生数量: 0
两者并集数量: 517
(occluded | dropout) == (mask == 0): True
```

几何遮挡连续段按每个历史窗口独立计算，共 47 段、总长 468：

| 指标 | JSON | 独立重算 | 差异 |
|---|---:|---:|---:|
| 样本数 | 100 | 100 | 0 |
| mask=1 / 可见比例 | 0.7415 | 0.7415 | 0 |
| 几何遮挡比例 | 0.234 | 0.234 | 0 |
| 随机丢帧比例 | 0.0245 | 0.0245 | 0 |
| 最大连续几何遮挡 | 19 | 19 | 0 |
| 平均连续几何遮挡 | 9.957446808510639 | 9.957446808510639 | 0 |

如果把所有缺失原因合并，最长连续缺失仍为 19，平均连续缺失为 `5.385416666666667`；JSON 中命名为 occlusion 的连续段明确采用 `history_occluded`，不是合并缺失。

数据格式能够区分真实树干遮挡与随机丢帧。当前生成逻辑只对几何可见帧抽样随机丢帧，所以两种原因互斥，重叠数为 0；该数值是真实计算结果，不是缺失统计的推断或伪造。

## 10. 可视化与数据一致性

### 10.1 验收发现与修复

初次验收发现：旧实现的三张图使用切窗前完整 201 帧序列，虽然属于同一次真实仿真，但没有对应到 NPZ 的某一行，也没有 scene/sample 标识，因此无法满足单样本核对要求。

已修复为：三图统一读取 `dataset` 的同一行，通过该行 `coordinate_origin` 恢复世界坐标；标题、日志和 `sample_statistics.json.visualization_example` 保存来源元数据。新增单元测试 `test_visualization_example_is_restored_from_one_dataset_row`。

### 10.2 最终示例元数据和计数

```text
scene_id: 0
sample_index: 50
sample_start_index: 60
history indices: [60, 79]
future indices:  [80, 99]

history 可见点: 8
history 几何遮挡点: 12
history 随机丢帧点: 0
history mask=0 总数: 12
```

与 NPZ 的 `dataset[50]` 对比：

```text
图使用的 history_mask 与 dataset[50].history_mask 完全一致: True
图使用的 history truth/observations 与 dataset[50] 恢复世界坐标后一致: True
图使用的 sensor/tree centers/radii 与 dataset[50] 一致: True
可见点计数 8 == dataset mask=1 计数 8: True
缺失点计数 12 == dataset mask=0 计数 12: True
遮挡点计数 12 == dataset history_occluded 计数 12: True
```

`scene_example.png` 显示该样本的 20 步历史和 20 步未来真值路径；`trajectory_example.png` 和 `mask_timeline.png` 显示同一样本的 20 步历史观测及 mask。

三张图片均已按原始分辨率实际打开检查：

- 非空白，场景、轨迹和 mask 色带清楚可见；
- 坐标轴使用 equal aspect，未发生 x/y 比例失真；
- 图例完整；
- 树干用世界坐标半径绘制，且 equal aspect 保证显示为圆，未发现半径显示错误；
- 标题明确包含 `scene 0 | sample 50 | history 60-79`；
- 图中的计数和 NPZ 第 50 行一致。

## 11. 已发现、已修复和尚未解决的问题

### 已发现的问题

1. 初版示例图画的是同一次仿真的完整原始序列，而不是可追溯到 `dataset.npz` 某一行的样本；无法报告单一 sample index 及与其 mask 对照。

### 已修复的问题

1. 三图改为统一从 `dataset` 第 50 行恢复世界坐标；
2. 图片标题增加 scene、sample 和历史索引；
3. `run.log` 与 `sample_statistics.json` 增加示例来源和真实点数；
4. README 明确闭线段/闭圆、相切及端点在圆内的遮挡定义；
5. 新增可视化来源回归测试，pytest 从 14 项增加到 15 项；
6. 修复后重新执行正式实验、完整 pytest 和三次复现性运行，全部退出状态为 0。

### 尚未解决的问题

- 无阻塞性代码或数据问题。
- 当前 smoke 配置只有一个 scene，因此多 scene 边界拼接属于本产物不可触发项；没有发现错误，也未在本次验收中增加多场景功能。
- 工作区尚未提交 Git；这是版本管理状态，不是实验运行故障。
- pip 的用户缓存目录不可写，pip 自动禁用缓存；不影响本次环境读取、实验或测试。

## 12. 是否可进入下一阶段

**可以。** 实验 1 的生成、遮挡、窗口、保存、统计、复现性、可视化可追溯性和测试均通过本报告所列实际检查。按照本次任务边界，本轮在此停止，未开始任何后续预测或控制模型实验。
