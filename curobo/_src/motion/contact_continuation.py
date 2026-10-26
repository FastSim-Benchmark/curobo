"""One paired terminal-contact goal with a locally continuous seed."""

import torch

from curobo._src.motion.motion_contact import boundary_contact_scope, contact_goal_candidates
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.time_budget import check_time_budget

from .axis_seeds import continuous_axis_seed, local_search_radius


def plan_contact_continuation(
    planner, target, current, holds, use_implicit_goal=True, *, diagnostics=None
):
    """Propose a nearby endpoint; bind all validation to that exact endpoint.

    The caller owns the original axis scope and deadline. This
    helper only accepts one paired target; it does not select or remap goals.
    """
    diagnostics = {} if diagnostics is None else diagnostics
    if target.num_goalset != 1:
        raise ValueError("Require one paired terminal target")
    state = current.reorder(planner.joint_names)
    bounds = planner.kinematics.get_joint_limits().position.clone()
    radius = local_search_radius(planner.kinematics, state.position[0])
    local = torch.stack(
        (
            torch.maximum(bounds[0], state.position[0] - radius),
            torch.minimum(bounds[1], state.position[0] + radius),
        )
    )
    iterator = None
    try:
        check_time_budget()
        planner.update_joint_limits(position=local)
        iterator = contact_goal_candidates(planner, target, state)
        endpoint = next(iterator, None)
    finally:
        try:
            if iterator is not None:
                iterator.close()
        finally:
            planner.update_joint_limits(position=bounds)
    diagnostics["endpoint_proposed"] = endpoint is not None
    if endpoint is None:
        return None
    q = endpoint.reorder(planner.joint_names).position
    if q.shape != state.position.shape or not bool(torch.isfinite(q).all()):
        raise ValueError("Invalid terminal endpoint")
    if bool(((q[0] < local[0]) | (q[0] > local[1])).any()):
        raise ValueError("Terminal endpoint escaped local bounds")
    endpoint = JointState.from_position(q.clone(), planner.joint_names)
    actual = planner.compute_kinematics(endpoint).tool_poses
    frames = list(target.tool_frames)
    positions = torch.stack([actual.get_link_pose(f).position[0] for f in frames])
    quaternions = torch.stack([actual.get_link_pose(f).quaternion[0] for f in frames])
    seed_target = GoalToolPose(
        frames, positions[None, None, :, None], quaternions[None, None, :, None]
    )
    diagnostics["endpoint_position_error_m"] = (
        torch.linalg.vector_norm(seed_target.position - target.position, dim=-1)
        .detach()
        .cpu()
        .tolist()
    )
    prepared = continuous_axis_seed(
        planner,
        seed_target,
        current,
        holds,
        terminal_proposal=lambda *args: endpoint,
        terminal_hint=endpoint,
        diagnostics=diagnostics.setdefault("continuation", {}),
    )
    diagnostics["continuous_seed_present"] = prepared is not None
    if prepared is None:
        return None
    _, final, path, record = prepared
    diagnostics["seed_record"] = record
    if not torch.equal(final, endpoint.position[0]):
        raise ValueError("Contact capture and path endpoint differ")
    with boundary_contact_scope(planner, current, endpoint, "end"):
        check_time_budget()
        checked = planner.ik_solver.solve_pose(
            target,
            current_state=current,
            seed_config=endpoint.position[:, None].repeat(
                1, planner.ik_solver.config.num_seeds, 1
            ),
            return_seeds=planner.trajopt_solver.config.num_seeds,
            run_optimizer=False,
        )
        diagnostics["endpoint_metrics_success"] = bool(checked.success.any())
        if not bool(checked.success.any()):
            return None
        finetune = planner.config.trajopt_finetune_attempts
        return planner.trajopt_solver.solve_pose(
            target,
            current,
            seed_config=endpoint.position[:, None].repeat(1, path.shape[1], 1),
            seed_traj=path,
            use_implicit_goal=use_implicit_goal,
            finetune_attempts=1 if finetune is None else finetune,
            finetune_dt_scale=0.55,
        )
