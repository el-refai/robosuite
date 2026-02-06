"""Post-hoc trajectory optimization for policy-learnable robot trajectories.

Takes a recorded sequence of joint configurations and re-optimizes for:
- Joint-space smoothness (velocity, acceleration, jerk minimization)
- High manipulability throughout the trajectory
- Cartesian-space motion minimization (smooth end-effector paths)
- Velocity limit enforcement
- Preservation of the original task-critical end-effector poses

This is intended to be run as a post-processing step after data collection,
before using the trajectories for policy training.
"""

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def optimize_trajectory(
    robot: pk.Robot,
    target_link_name: str,
    joint_trajectory: onp.ndarray,
    dt: float = 0.02,
    smoothness_weight: float = 1.0,
    acceleration_weight: float = 0.5,
    jerk_weight: float = 0.1,
    manipulability_weight: float = 0.3,
    trajectory_tracking_weight: float = 5.0,
    cartesian_smoothness_weight: float = 2.0,
    velocity_limit_weight: float = 10.0,
    endpoint_weight: float = 100.0,
) -> tuple[onp.ndarray, dict]:
    """Post-optimize a recorded joint trajectory for policy learnability.

    Re-solves the trajectory as a batch optimization problem, balancing
    smoothness, manipulability, and fidelity to the original end-effector
    path.  The result is a trajectory that is easier for a downstream
    policy network to imitate.

    Args:
        robot: PyRoKi Robot.
        target_link_name: End-effector link name.
        joint_trajectory: (T, num_actuated_joints) recorded joint configs.
        dt: Timestep between configurations (seconds).
        smoothness_weight: Weight for joint-space velocity smoothness.
        acceleration_weight: Weight for acceleration minimization.
        jerk_weight: Weight for jerk minimization.
        manipulability_weight: Weight for manipulability maximization.
        trajectory_tracking_weight: Weight for tracking original trajectory
            in joint space (prevents excessive deviation).
        cartesian_smoothness_weight: Weight for Cartesian EE displacement
            minimization between consecutive steps.
        velocity_limit_weight: Weight for enforcing joint velocity limits.
        endpoint_weight: Weight for pinning start/end configurations.

    Returns:
        Tuple of (optimized_trajectory, metrics) where optimized_trajectory
        is (T, num_actuated_joints) and metrics is a dict with before and
        after quality metrics.
    """
    T, n_joints = joint_trajectory.shape
    if T < 4:
        return joint_trajectory.copy(), {"skipped": True, "reason": "too short"}

    target_link_index = robot.links.names.index(target_link_name)
    init_traj = jnp.array(joint_trajectory)

    # --- Set up trajectory variables ---
    traj_vars = robot.joint_var_cls(jnp.arange(T))
    robot_batched = jax.tree.map(lambda x: x[None], robot)

    factors: list[jaxls.Cost] = []

    # 1. Joint-space trajectory tracking
    factors.append(
        pk.costs.rest_cost(
            traj_vars,
            init_traj,
            jnp.array(trajectory_tracking_weight),
        )
    )

    # 2. Joint limits
    factors.append(
        pk.costs.limit_cost(
            robot_batched,
            traj_vars,
            jnp.array([100.0])[None],
        )
    )

    # 3. Endpoint constraints
    start_cfg = init_traj[0]
    end_cfg = init_traj[-1]
    n_pin = min(2, T)
    factors.extend(
        [
            jaxls.Cost(
                _endpoint_residual,
                (
                    robot.joint_var_cls(jnp.arange(0, n_pin)),
                    start_cfg,
                    endpoint_weight,
                ),
                name="start_constraint",
            ),
            jaxls.Cost(
                _endpoint_residual,
                (
                    robot.joint_var_cls(jnp.arange(max(0, T - n_pin), T)),
                    end_cfg,
                    endpoint_weight,
                ),
                name="end_constraint",
            ),
        ]
    )

    # 4. Joint-space smoothness
    if T >= 2:
        factors.append(
            pk.costs.smoothness_cost(
                robot.joint_var_cls(jnp.arange(1, T)),
                robot.joint_var_cls(jnp.arange(0, T - 1)),
                jnp.array([smoothness_weight])[None],
            )
        )

    # 5. Velocity limit enforcement
    if T >= 2 and velocity_limit_weight > 0:
        factors.append(
            pk.costs.limit_velocity_cost(
                robot_batched,
                robot.joint_var_cls(jnp.arange(1, T)),
                robot.joint_var_cls(jnp.arange(0, T - 1)),
                dt=dt,
                weight=velocity_limit_weight,
            )
        )

    # 6. Acceleration minimisation (5-point stencil, requires T >= 5)
    if T >= 5 and acceleration_weight > 0:
        factors.append(
            pk.costs.five_point_acceleration_cost(
                robot.joint_var_cls(jnp.arange(2, T - 2)),
                robot.joint_var_cls(jnp.arange(4, T)),
                robot.joint_var_cls(jnp.arange(3, T - 1)),
                robot.joint_var_cls(jnp.arange(1, T - 3)),
                robot.joint_var_cls(jnp.arange(0, T - 4)),
                dt,
                jnp.array([acceleration_weight])[None],
            )
        )

    # 7. Jerk minimisation (7-point stencil, requires T >= 7)
    if T >= 7 and jerk_weight > 0:
        factors.append(
            pk.costs.five_point_jerk_cost(
                robot.joint_var_cls(jnp.arange(6, T)),
                robot.joint_var_cls(jnp.arange(5, T - 1)),
                robot.joint_var_cls(jnp.arange(4, T - 2)),
                robot.joint_var_cls(jnp.arange(2, T - 4)),
                robot.joint_var_cls(jnp.arange(1, T - 5)),
                robot.joint_var_cls(jnp.arange(0, T - 6)),
                dt,
                jnp.array([jerk_weight])[None],
            )
        )

    # 8. Manipulability maximisation at each timestep
    if manipulability_weight > 0:
        factors.append(
            pk.costs.manipulability_cost(
                robot_batched,
                traj_vars,
                target_link_indices=jnp.array(
                    target_link_index, dtype=jnp.int32
                ),
                weight=manipulability_weight,
            )
        )

    # 9. Cartesian-space smoothness
    if T >= 2 and cartesian_smoothness_weight > 0:
        factors.append(
            jaxls.Cost(
                _cartesian_smoothness_residual,
                (
                    robot_batched,
                    robot.joint_var_cls(jnp.arange(1, T)),
                    robot.joint_var_cls(jnp.arange(0, T - 1)),
                    jnp.array(target_link_index, dtype=jnp.int32),
                    cartesian_smoothness_weight,
                ),
                name="CartesianSmoothness",
            )
        )

    # --- Solve ---
    solution = (
        jaxls.LeastSquaresProblem(factors, [traj_vars])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                (traj_vars.with_value(init_traj),)
            ),
            verbose=False,
        )
    )
    optimized = onp.array(solution[traj_vars])

    # --- Compute before / after metrics ---
    metrics_before = compute_trajectory_metrics(
        robot, target_link_name, joint_trajectory
    )
    metrics_after = compute_trajectory_metrics(
        robot, target_link_name, optimized
    )
    metrics = {"before": metrics_before, "after": metrics_after}

    return optimized, metrics


# ---------------------------------------------------------------------------
# Internal residual helpers
# ---------------------------------------------------------------------------


def _endpoint_residual(
    vals: jaxls.VarValues,
    var: jaxls.Var[jax.Array],
    target_cfg: jax.Array,
    weight: float,
) -> jax.Array:
    """Pin trajectory endpoints to a fixed configuration."""
    return ((vals[var] - target_cfg) * weight).flatten()


def _cartesian_smoothness_residual(
    vals: jaxls.VarValues,
    robot: pk.Robot,
    curr_var: jaxls.Var[jax.Array],
    prev_var: jaxls.Var[jax.Array],
    target_link_index: jax.Array,
    weight: float,
) -> jax.Array:
    """Penalise large Cartesian EE displacements between consecutive steps."""
    curr_fk = robot.forward_kinematics(vals[curr_var])
    prev_fk = robot.forward_kinematics(vals[prev_var])
    curr_pos = jaxlie.SE3(curr_fk[..., target_link_index, :]).translation()
    prev_pos = jaxlie.SE3(prev_fk[..., target_link_index, :]).translation()
    return ((curr_pos - prev_pos) * weight).flatten()


# ---------------------------------------------------------------------------
# Trajectory quality metrics
# ---------------------------------------------------------------------------


def compute_trajectory_metrics(
    robot: pk.Robot,
    target_link_name: str,
    joint_trajectory: onp.ndarray,
) -> dict:
    """Compute trajectory quality metrics for policy-learnability assessment.

    Args:
        robot: PyRoKi Robot.
        target_link_name: End-effector link name.
        joint_trajectory: (T, num_actuated_joints) joint configurations.

    Returns:
        Dictionary with rms_joint_velocity, rms_joint_acceleration,
        cartesian_path_length, mean_manipulability, min_manipulability,
        and joint_smoothness.
    """
    T = joint_trajectory.shape[0]
    traj = jnp.array(joint_trajectory)
    target_link_index = robot.links.names.index(target_link_name)

    metrics: dict = {}

    # Joint-space velocity
    if T >= 2:
        joint_vel = jnp.diff(traj, axis=0)
        metrics["rms_joint_velocity"] = float(
            jnp.sqrt(jnp.mean(joint_vel ** 2))
        )
        metrics["joint_smoothness"] = float(jnp.sum(jnp.abs(joint_vel)))
    else:
        metrics["rms_joint_velocity"] = 0.0
        metrics["joint_smoothness"] = 0.0

    # Joint-space acceleration
    if T >= 3:
        joint_acc = jnp.diff(traj, n=2, axis=0)
        metrics["rms_joint_acceleration"] = float(
            jnp.sqrt(jnp.mean(joint_acc ** 2))
        )
    else:
        metrics["rms_joint_acceleration"] = 0.0

    # Cartesian EE path length and manipulability (per-timestep)
    ee_positions = []
    manip_values = []
    for t in range(T):
        fk = robot.forward_kinematics(traj[t])
        pose = jaxlie.SE3(fk[target_link_index])
        ee_positions.append(onp.array(pose.translation()))

        # Yoshikawa manipulability via translation Jacobian
        jacobian = jax.jacfwd(
            lambda q: jaxlie.SE3(
                robot.forward_kinematics(q)
            ).translation()
        )(traj[t])[target_link_index]
        JJT = jacobian @ jacobian.T
        m = jnp.sqrt(jnp.maximum(0.0, jnp.linalg.det(JJT)))
        manip_values.append(float(m))

    ee_positions_arr = onp.stack(ee_positions)
    if T >= 2:
        deltas = onp.diff(ee_positions_arr, axis=0)
        metrics["cartesian_path_length"] = float(
            onp.sum(onp.linalg.norm(deltas, axis=1))
        )
    else:
        metrics["cartesian_path_length"] = 0.0

    manip_arr = onp.array(manip_values)
    metrics["mean_manipulability"] = float(onp.mean(manip_arr))
    metrics["min_manipulability"] = float(onp.min(manip_arr))

    return metrics
