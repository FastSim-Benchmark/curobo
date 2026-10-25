# 规划器运动质量研究工作区

维护日期：2026-10-23。适用目录：`/home/ubuntu/desktop/fastsim-planner-research`。

## 修改授权与来源

- 用户明确要求创建独立项目，后续直接修改规划器。本项目允许修改 `curobo/` 中的
  规划、IK、候选生成/排序、轨迹优化、相关代价/约束，以及 `curobo/tests/`、
  `research/scripts/`、实验配置和文档；不受旧项目“只能改Harness”的范围限制。
- 例行读代码、修复、回归及新建实验不重复请求许可。不按案例ID、固定关节角或目标写生产特例。
- 不修改原配置项目 `/home/ubuntu/desktop/jd_fastsim-task-preparation`、共享依赖
  `/home/ubuntu/desktop/fastsim_benchmark_project_20260902/curobo`，或其FastSim、
  FastSim-Plugins、LayoutGen、task2sim。后续执行与修改入口统一使用本项目。
- 本项目从用户本地 `/home/ubuntu/desktop/fastsim_benchmark_project_20260902/curobo`
  独立克隆，基线 `30dd1194b799498699a8cf4aa8b0d5412a777d15`，研究分支
  `codex/planner-natural-motion`。没有共享worktree、符号链接或Git对象库依赖。
- 本地这份代码的来源是 `https://github.com/FastSim-Benchmark/curobo` 项目分支，
  不是NVIDIA原版cuRobo，也不是FastSim Core。不能把“从本地复制”描述成与该组织无关。
- 建项目时没有重新拉远端更新。查上游时记录日期、仓库、分支、SHA、文件；远端能力和
  本机实测能力分开。不自动pull、切换基线或用其他NVIDIA checkout替代本分支。
- 默认修改仅在本地。明确要求commit才提交，明确要求push才推送，不自动合并/发布。
  不覆盖未提交内容，不做破坏性reset/checkout。完整来源见 `research/provenance/bootstrap.json`。

## 当前问题与已验证结论

研究空场景中的多余关节动作、抬肘、TCP绕行，以及同一请求受前序调用影响的问题。
机器人为G2 OP3-E，单臂7、双臂14自由度；双臂目标向同一次原生plan_pose传入。
原生TCP为 `left_gripper_center`、`right_gripper_center`。保持185个原生碰撞球、
原生自碰撞忽略表、真实关节限制及非活动关节锁定。实验速度上限0.4rad/s，其余按manifest。

| 回归案例ID（仅用于测试） | 原问题 | 关闭提前返回后的完整顺序实验 |
| --- | --- | --- |
| `both_nearby_2` | 左手起终点约7.72cm，路径85.93cm，抬肘39.25cm | 左手路径7.83cm，抬肘接近0 |
| `both_roll_positive` | 终点翻腕90度，左手路径3.059m，超出两端的抬肘16.41cm | 左手38.85cm，但右手从1.86cm增长到27.29cm |
| `right_restore_2` | 右手回正路径61.71cm，抬肘17.70cm | 路径24.88cm，抬肘1.29cm |

- 代码已有起点距离排序：LM的start_cspace_dist_weight=0.01，主IK使用
  start_cspace_dist_tolerance。不能断言“完全没有近邻偏好”。
- 实测：近邻候选未达精度，远候选先通过，`exit_early=True`跳过后续优化器。
  小动作2近邻移动约0.139rad但误差6.11mm，超过5mm容差；远处成功候选约3.939rad。
  固定原远终点重新plan_cspace仍绕远，主要问题已在终点构型选择阶段形成。
- 固定random_seed=5只固定初始状态，不会每次重置样本；LM默认采样seed另为451。
  只向全新规划器移植历史LM采样器状态，三例的LM及主IK返回候选逐元素一致。
  完整轨迹仍有微小差异，不能据此宣布所有缓存/数值状态均无影响。
- 已执行的改法是诊断脚本refine分支设置 `planner.ik_solver.config.exit_early=False`。
  建项目时没有把它写成规划器生产默认值。
- 原48个pose动作36成功/12未得解；完整顺序refine的成功失败名单不变。但独立调用仍
  敏感，翻腕时右手路径可达81.55cm。取消提前返回只是第一步，不能称为完整修复。
- `joints`是精确关节见证对照，`same_endpoint`固定原pose选中的终点。refine只接收
  原TCP目标，不使用见证关节角。不能把见证对照冒充算法改善。
- 普通pose请求仅约束终点；“原地翻腕”案例未自动设置沿途TCP保持硬约束。
- 证据是数值规划和真实网格的运动学回放，没有Mission/物理执行。筛查标记归零不等于
  自然性验收，人工审核pending。历史解释需独立核验，不作为免检前提。

## 文件与证据入口

以下路径相对本项目根目录。优先使用这里的独立副本：

| 内容 | 路径 |
| --- | --- |
| 当前可改规划器 | `curobo/_src/motion/motion_planner.py` |
| IK提前返回/收敛/排序 | `curobo/_src/solver/solver_ik.py`、`solver_ik_cfg.py` |
| LM候选 | `curobo/_src/solver/seed_ik/seed_ik_solver.py`、`seed_ik_solver_cfg.py` |
| 采样状态 | `curobo/_src/util/sampling/sample_buffer.py`、`sequencer_halton.py` |
| TrajOpt与排名 | `curobo/_src/solver/solver_trajopt.py`、`solver_trajopt_result.py` |
| 原生代价配置 | `curobo/content/configs/task/ik/`、`trajopt/` |
| 当前诊断与回归工具 | `research/scripts/` |
| 冻结48例及机器人配置 | `research/fixtures/empty_world_v1/` |
| 三例回归定义 | `research/fixtures/arm_motion_detour_regressions.json` |
| 详细分析/反例 | `research/evidence/gpt-review-20261023/reports/问题分析与改善.md` |
| 原始三例轨迹与对照 | `research/evidence/gpt-review-20261023/evidence/baseline/` |
| 48例改法比较 | `research/evidence/gpt-review-20261023/evidence/refinement/algorithm-comparison.json` |
| 15次诊断/trace/随机状态 | `research/evidence/gpt-review-20261023/evidence/diagnostics/` |
| 完整视频与12个动画索引 | `research/evidence/gpt-review-20261023/human-review/INDEX.md` |
| 给GPT的审查问题 | `research/evidence/gpt-review-20261023/START_HERE_给GPT.md` |
| 原ZIP | `research/archive/FastSim_Planner_GPT_Review_20261023.zip` |
| 独立机器人URDF/mesh/配置 | `research/assets/g2_crsb_op3_e/` |
| 复制资产hash | `research/provenance/robot-assets-sha256.json` |
| 新实验输出 | `runs/<new_run>/` |

历史证据的绝对路径、原源码与hash保留原样。新run从冻结输入重定位并重新记录hash，不改写
历史证据。证据包source/是历史副本，不能当当前实现或加入PYTHONPATH；不得导入旧task_harness。
原上游AGENTS/README保留在 `research/provenance/upstream-AGENTS.md`、`upstream-README.md`。

## 环境、运行与验收

- 先确认pwd/Git顶层为本项目，检查git status，读本文件、README及本次相关源码/测试。
- 算法遵守 `.agent/design_principles.mdc`；Python参考python_rules/package_format_rules，
  测试参考tests，CUDA/C++参考cpp_rules。小步实现，类型和张量形状明确，内部从具体模块
  导入，正常运行保留CUDA graph。用户最新要求优先于通用开发规则。
- GPU解释器：`/home/ubuntu/miniforge3/envs/fastsim_vnext/bin/python`。
  分析/USD解释器：`/home/ubuntu/miniforge3/envs/isaac5/bin/python`。RTX5090曾已验证可用，
  每次以实际CUDA探测为准；不更改系统/base或其他项目的共享环境安装。
- GPU运行显式 `PYTHONPATH=/home/ubuntu/desktop/fastsim-planner-research`，核对
  `curobo.__file__`。fastsim_vnext的历史editable安装曾指向另一份NVIDIA checkout，
  不能依赖默认导入。渲染使用isaac5、`PYOPENGL_PLATFORM=egl`和本项目`.runtime/visual/`。
- 新run记录输入、版本、未提交源码内容hash、完整顺序、候选/随机状态、约束和耗时。
  运行中不改代码/输入；前后hash一致才接受结果。新改动必须新建run，不覆盖旧数据。
- 长任务监管到退出，每60秒以内检查进程、日志更新时间及进度。预算有限，区分未调用、
  未得解、异常和超时。不启动后结束对话。主动停止活跃实验先征求用户同意；事前声明的
  有限超时和自然结束除外，确认资源不足即将失去响应时可停止必要进程并说明。
- 行为修改添加有意义的回归，包含问题案例、正常反例、完整调用顺序和独立调用；
  后续加入障碍、奇异和关节边界，验证全局回退。不放宽碰撞、容差或关节限制换成功率。
- 同时报告每只手TCP路径/偏移、实际关节转角、肘部超出两端的抬升、终点误差、成功率和
  稳态耗时。不能只报告总关节距离或一只手，不能把未得解写成物理不可达。
- 交付本地WebM/视频并优先系统播放器，不生成网页。USD打包遵守
  `research/docs/usd-delivery-style.md`，全部主入口统一human-review/并提供完整索引。
- runs、证据、原始资产、运行时库和ZIP默认不提交。只读解释器目前共享，源码和机器人
  数据在本项目独立复制；未复制原始机器人USD/物理资产，不能宣称完整物理环境已迁移。

## 后续调试提示词

> 追踪本项目LM初值、IK近邻候选、提前返回、优化目标与选解排序、TrajOpt初始化和最终
> 排名，独立检查现有因果解释。用最少的受控实验区分采样、缓存和优化状态；不要只靠
> 重置随机数掩盖质量问题。保持真实碰撞、精度和关节边界。

> 以三例问题和48例动作为基线，设计稳定的局部近邻精修与有界全局回退。允许直接修改
> 本项目规划器及对应测试。区分实际改动、实测结果、待验证建议；不按案例ID写特例，
> 不把精确见证作为pose规划器的隐藏答案。

> 设计同时考虑双手TCP偏移、肘部额外抬升、真实关节转角和时间的目标/排名策略。
> 检查是否把绕行从一手转移到另一手。只有任务明确要求时才添加沿途保持/直线硬约束，
> 普通pose目标保留原有语义。

> 对改动运行问题回归、正常反例、完整顺序与全新规划器独立调用，记录源码/输入hash、
> 成功率、终点误差、路径质量、顺序稳定性和稳态耗时，验证失败回退。不要把运动学回放
> 当物理执行，也不要把筛查阈值通过当自然性完全通过。

## 上游通用开发规则（保留）

# Agent Instructions

This file is the entry point for coding agents working in this repository. Keep
changes small, inspect the existing code before editing, and follow the
repository-specific rules in `.agent/`.

## Rule Files

- Read `.agent/design_principles.mdc` before non-trivial code changes.
- Use `.agent/python_rules.mdc` and `.agent/package_format_rules.mdc` for Python
  code.
- Use `.agent/cpp_rules.mdc` for C++, CUDA, and Warp-adjacent native code.
- Use `.agent/tests.mdc` before adding or changing tests.
- Use `.agent/sphinx_md.mdc`, `.agent/documenter.mdc`, and
  `.agent/diataxis.mdc` for documentation changes.
- Use `.agent/file_naming.mdc` when creating or moving implementation files.

## Development Workflow

- Prefer `rg`/`rg --files` for repository search.
- Activate `.venv` before running Python commands when it exists.
- Preserve user changes in the working tree. Do not revert unrelated files.
- Keep public APIs minimal and user-focused. Do not expose implementation details
  unless they are required by the user-facing contract.
- Prefer existing local patterns and helpers over new abstractions.
- Use only `Copyright (c) 2026` in SPDX copyright headers for new files;
  preserve existing headers when modifying older files.
- Add focused tests for behavioral changes. Do not use `float64` in tests.
- Keep CUDA graph enabled for normal runtime, examples, and performance-sensitive
  paths. Disable CUDA graph only in tests or debugging paths where eager
  execution helps catch correctness issues.

## Python Conventions

- Add type hints to all function signatures.
- Use concise docstrings that describe behavior, inputs, outputs, and tensor
  shapes when relevant.
- For dataclasses, document fields where they are declared.
- Use logging/error helpers from `curobo/_src/util/logging.py` instead of raw
  prints or ad hoc exceptions in library code.
- Keep `_src` package initializers marker-only unless they intentionally build a
  public API aggregation layer.
- Public modules under `curobo/` should curate user-facing imports. Internal
  code should import concrete implementation modules directly.

## Naming

- Internal implementation files under `curobo/_src/` should use the
  category-first pattern from `.agent/file_naming.mdc`.
- Public API modules under `curobo/` should use descriptive, user-friendly names.

## Documentation

- Prefer Markdown with MyST syntax for new Sphinx documentation.
- Keep docs in the correct Diataxis category: tutorial, guide, concept, or
  reference.
- Write user-facing docs around the outcome and contract, not internal
  implementation details.
