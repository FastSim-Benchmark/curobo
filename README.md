# FastSim 规划器运动质量研究

本项目直接研究和修改基于cuRobo的规划器：候选IK、关节终点选择、轨迹优化、多余动作与调用顺序敏感性。工作规则见 [AGENTS.md](AGENTS.md)。

源码独立复制自本地FastSim-Benchmark/curobo，基线commit `30dd1194b799498699a8cf4aa8b0d5412a777d15`，分支 `codex/planner-natural-motion`。上游README原文保存在 [research/provenance/upstream-README.md](research/provenance/upstream-README.md)。

## 从哪里继续

- 当前实现：`curobo/_src/motion/`、`curobo/_src/solver/`、`curobo/_src/cost/`。
- 当前问题、已知反例及后续提示词：[AGENTS.md](AGENTS.md)。
- 原三例、48例对照与15次诊断：[证据阅读入口](research/evidence/gpt-review-20261023/START_HERE_给GPT.md)。
- 已有视频和12个机器人动画：[完整索引](research/evidence/gpt-review-20261023/human-review/INDEX.md)。
- 数据/版本来源：[bootstrap.json](research/provenance/bootstrap.json)。

默认`trace`是原始算法带追踪；`refine`仅实验性关闭IK提前返回。其收益与反例已保存，尚未设为规划器默认值。

## 启用运动质量策略

规划器新增可选策略，普通pose请求仍只约束终点：

```python
config = MotionPlannerCfg.create(
    robot=robot_config,
    prefer_smooth_motion=True,
    quality_refinement_iters=20,
)
planner = MotionPlanner(config)
```

该策略保留32个主IK候选，为近起点初值保留名额；已有可行终点但动作质量不足时做短精修，
没有可行终点时保留原优化预算。对活动关节互不共享的多TCP，补充可复现的逐工具局部IK提案，
组合后重新执行原生全身检查。终点草稿和轨迹分别评价每只手的额外平移、旋转路程及实际关节运动。
原始全局初值会保留，新轨迹失败时优先恢复原选解流程；回退计入原max_attempts和超时预算。
CUDA Graph、原生碰撞模型、关节限制和精度要求继续生效。

兼容既有调用时保留默认`prefer_smooth_motion=False`。详细失败几何诊断默认关闭，需调试时设置
`config.ik_solver_config.detailed_failure_diagnostics=True`；原生约束判断不受该开关影响。

新的冻结输入对照入口为`research/scripts/benchmark_motion_quality.py`，例如：

```bash
PYTHONPATH="$PWD" /home/ubuntu/miniforge3/envs/fastsim_vnext/bin/python \
  research/scripts/benchmark_motion_quality.py \
  --run runs/new_motion_quality_comparison --policies legacy quality20 --repeats 3
```

加入`--trace-candidates`保存候选，加入`--independent`使用全新规划器逐例测试。候选追踪与纯性能
统计应分别运行，输出保留源码快照、输入与源码hash、调用次序和采样器状态。

## 在本项目运行回归

使用独立复制的机器人URDF、mesh、碰撞球和冻结输入，输出写入本项目全新runs目录。原数据无需改写。

```bash
cd /home/ubuntu/desktop/fastsim-planner-research
PYTHONPATH="$PWD" /home/ubuntu/miniforge3/envs/fastsim_vnext/bin/python \
  research/scripts/run_regression.py run \
  --run runs/example_both_trace_r01 --group both --policy trace
```

`--case both_nearby_2`表示单个独立调用；省略`--case`按该组原顺序运行16个动作。`--group`可选left/right/both。每例依次进行pose、joints见证对照、same_endpoint对照；某项缺少可用前提时明确跳过。

修改策略另建run：

```bash
PYTHONPATH="$PWD" /home/ubuntu/miniforge3/envs/fastsim_vnext/bin/python \
  research/scripts/run_regression.py run \
  --run runs/example_both_refine_r01 --group both --policy refine
```

规划使用fastsim_vnext，CPU分析自动使用isaac5。单组规划预先限制240秒，分析120秒；单次原生请求最多3次尝试、60秒。父进程持续等待并每30秒以内报告存活状态。算法改动期间不要修改活跃run的输入或源码。

`inputs/study.json`和`end-validation.json`记录实际导入路径、源码及独立资产hash；`analysis.json`给出实际路径指标。历史证据中绝对路径只保留来源，不是当前执行入口。所有实验代码在research/scripts，本项目不导入task_harness。

## 本地视频与USD

```bash
PYOPENGL_PLATFORM=egl PYTHONPATH="$PWD/.runtime/visual" \
  /home/ubuntu/miniforge3/envs/isaac5/bin/python \
  research/scripts/render_empty_arm_motion.py --run runs/<已完成run>
```

渲染使用本项目复制的依赖，输出本地WebM和human-review索引。不要制作网页。单组完整顺序run可能生成很多视频，开始前先明确本轮展示子集，避免无必要的长渲染。USD交付规范见 [usd-delivery-style.md](research/docs/usd-delivery-style.md)。

源码、研究脚本可纳入后续用户授权的提交；runs、证据包、机器人资产、运行时库和ZIP均默认忽略。原机器人USD/物理环境没有迁移，本项目目前用于数值规划和运动学回放。
