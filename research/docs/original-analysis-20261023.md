# 空场景绕路案例：选解原因与第一轮改善（2026-10-23）

三个问题案例已冻结在 [arm_detour_baseline_20261023_r01](/home/ubuntu/desktop/jd_fastsim-task-preparation/task-yaml-harness/runs/arm_detour_baseline_20261023_r01)。完整输入、原始轨迹、近邻关节见证、固定原终点对照、隔离复测及原调用顺序均保留；human-review 含9个完整机器人动画入口和3个原始对照视频。回归定义另存于 [arm_motion_detour_regressions.json](/home/ubuntu/desktop/jd_fastsim-task-preparation/task-yaml-harness/config/arm_motion_detour_regressions.json)。所有源文件和产物均为本地文件，未提交或推送。

## 使用的真实代码与边界

FastSim-Benchmark/curobo，main，本机 commit `30dd1194b799498699a8cf4aa8b0d5412a777d15`；源码位于 `/home/ubuntu/desktop/fastsim_benchmark_project_20260902/curobo`。使用当前 G2_0915/g2_crsb_op3_e 原生机器人，单臂7自由度，双臂14自由度；双臂目标在同一次原生 plan_pose 中求解。原生185个碰撞球、自碰撞检查、非活动关节锁定和关节限制保留，速度上限0.4rad/s，CUDA graph开启。本次为数值规划与真实网格的运动学回放，未执行 Mission 或物理仿真。

## 结论：远构型先通过，近构型尚未精修

不是简单缺少“离起点近”的排序。LM阶段已经有 start_cspace_dist_weight=0.01；主IK的 `_get_result` 同时计入 start_cspace_dist_tolerance 和末端误差。追踪显示，传给轨迹优化的4个成功候选中最近者已经很远：双臂小动作2为3.939rad、双臂正向翻腕为4.943rad、右臂回正2为2.934rad。

从当前关节出发的近邻候选仍在32个主IK输入候选中，但未达收敛要求。双臂小动作2的近邻只移动0.139rad，却还有约6.11mm位置误差（默认容差5mm）；另两个案例的近邻位置误差约71mm、75mm。远处候选先满足精度和碰撞要求，`IKSolverCfg.exit_early=True` 使 `_solve_impl` 提前返回，跳过后续优化器。轨迹优化接到的是远终点；固定该终点重规划仍然绕远，说明问题主要已在终点选择阶段形成。

源码依据（以上 commit 的文件副本及hash另存于 inputs/native-source-evidence）：

- [solver_ik.py](/home/ubuntu/desktop/fastsim_benchmark_project_20260902/curobo/curobo/_src/solver/solver_ik.py:391)：先检查种子指标，满足条件则提前返回；否则运行优化器。
- [solver_ik.py](/home/ubuntu/desktop/fastsim_benchmark_project_20260902/curobo/curobo/_src/solver/solver_ik.py:450)：按成功状态、起点关节距离与末端误差选解。
- [seed_ik_solver.py](/home/ubuntu/desktop/fastsim_benchmark_project_20260902/curobo/curobo/_src/solver/seed_ik/seed_ik_solver.py:523)：LM候选已有起点距离排序。
- [solver_ik_cfg.py](/home/ubuntu/desktop/fastsim_benchmark_project_20260902/curobo/curobo/_src/solver/solver_ik_cfg.py:74)：exit_early 默认开启。
- [lbfgs_ik.yml](/home/ubuntu/desktop/fastsim_benchmark_project_20260902/curobo/curobo/content/configs/task/ik/lbfgs_ik.yml:14)：优化器的 cspace_target_weight 为0；这与结果排序中的起点距离不是同一个目标项。

这些是本轮三个案例的实测原因，不代表所有抬肘动作都不合理，也不证明全局最优路径已经找到。

## 调用顺序为何改变结果

同一规划器跨调用保留 LM 的 Halton样本缓存与随机索引生成器状态。固定 random_seed=5 固定的是初始状态，不会让每次请求使用同一批样本；LM采样器另有默认 sampler_seed=451。先前的pose调用及重试会消耗样本，改变后续候选。每个案例的机器人起点都明确重置，因此不是机器人沿用上一次终点造成的。

受控证据：只把完整历史之后的 LM 随机生成器状态移植到全新规划器，三个案例的LM候选和主IK返回候选均逐元素完全一致；完整轨迹最大差异分别约0.000167、0.0000216、0.0000715rad，因此不把最终轨迹称为逐点完全一致。保留暖缓存、仅在请求前重置LM采样器，也使小动作及回正接近独立调用的结果；翻腕仍然较差。采样状态足以解释本轮选解分支，不能据此断言缓存对所有数值过程绝无影响。[受控结果](/home/ubuntu/desktop/jd_fastsim-task-preparation/task-yaml-harness/runs/arm_detour_refinement_review_20261023_r02/causal-checks.json)。

## 第一轮改善：让已有近邻候选继续收敛

本地诊断策略仅设置 `planner.ik_solver.config.exit_early=False`，不修改上游文件，不使用见证关节角作为改善输入，不放宽任何原生限制或容差。全部48个动作使用同一策略。

| 案例与观察手 | 原TCP路径 | 实验TCP路径 | 原关节路径 / 实验关节路径 |
|---|---:|---:|---:|
| 双臂附近小动作2，左手 | 85.93cm | 7.83cm | 3.951 / 0.225rad |
| 双臂正向翻腕90度，左手 | 305.90cm | 38.85cm | 5.420 / 3.925rad |
| 右臂回正2，右手 | 61.71cm | 24.88cm | 2.944 / 0.773rad |

小动作2左肘上升约39.25cm降为接近0；右臂回正抬肘约17.70cm降为1.29cm。翻腕左肘超出两端的额外抬升16.41cm消除，但整个动作仍非最优：右手TCP路径从1.86cm增长到27.29cm，说明总关节距离较短不足以保证双手各自路径自然。翻腕请求只约束末端终点，并未要求沿途TCP原地不动，不能把中途偏移表述为违反已设置的硬约束。

48个pose请求仍为36成功/12未得解，名单完全一致；含两类关节对照，实际调用123次：105成功、18未得解、21个对照未调用。没有新增动作覆盖或API费用。36个成功pose结果中最大终点位置误差约4.21mm、角度误差约0.0306rad，符合本轮原生容差；三个重点案例的误差更小。稳态调用耗时大致由约45–50ms增加至55–65ms，这只是含追踪开销的本轮观察，不是正式性能benchmark；首轮JIT耗时不参与比较。

48例筛查标记由5例降为0只表示既定阈值不再触发，不是自然性验收，尤其原地翻腕需要单独的TCP位移指标。[完整48例对比](/home/ubuntu/desktop/jd_fastsim-task-preparation/task-yaml-harness/runs/arm_detour_refinement_review_20261023_r02/algorithm-comparison.json)。

## 改善后的顺序稳定性仍未通过

另外新建三个全新规划器，以相同实验策略独立调用。小动作2左手路径约9.33cm、右臂回正约25.28cm，仍比原始顺序中的绕路短；但关节路径与完整顺序仍不同。双臂翻腕左手约36.51cm，右手却达到81.55cm，总关节路径5.681rad。因此取消提前返回只是已验证的第一步，不能直接作为完整修复。后续需让局部近邻候选的生成/收敛更稳定，并在轨迹评分中同时照顾两只手。[顺序稳定性实测](../runs/arm_detour_refinement_review_20261023_r02/refined-order-check.json)。

本轮全部诊断包含15个受控运行、387次实际规划调用（333成功、54本轮未得解）、63个对照未调用；覆盖的独立动作仍是48个。原始基线和视频派生副本不重复计数，未产生API费用。所有子进程已结束。

## 后续算法方向

1. 在接受远解前，为当前姿态附近的可行候选保留精修预算；首先比较“总是精修”和“质量触发后精修”的成功率与耗时，不能只靠重置随机数掩盖问题。
2. 每个请求的局部候选保持稳定，同时保留全局候选作为逃离局部极值的后备；单独记录采样器状态，保留完整顺序和孤立调用两套回归。
3. 对通过碰撞和精度检查的多条轨迹，再比较每只手的TCP偏移、关节实际转角、肘部额外抬升和动作时长。双臂需避免以一侧改善换另一侧大幅绕行。
4. 如果任务确实要求原地翻腕、直线回撤或另一只手保持，应明确传入路径约束；不要把所有一般pose任务强制为直线，也不按案例ID添加生产分支。

本轮完成原因验证与本地改善原型，尚未把它写入共享上游的正式算法。之后的正式修改应以本基线、48例正常/异常回归及新增障碍场景共同验收。人工复核pending。
