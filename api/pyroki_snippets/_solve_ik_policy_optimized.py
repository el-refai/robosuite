"""Policy-optimized IK: combines pose matching, joint continuity, and
manipulability maximization for smoother, more learnable trajectories.

This solver is designed to produce joint configurations that are:
- Smooth (low velocity between consecutive calls via continuity cost)
- High manipulability (favors well-conditioned configurations)
- Accurate (precise pose matching for the end-effector)
"""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def solve_ik_policy_optimized(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: onp.ndarray,
    target_position: onp.ndarray,
    prev_cfg: onp.ndarray,
    initial_cfg: onp.ndarray | None = None,
    continuity_weight: float = 0.1,
    manipulability_weight: float = 0.5,
) -> onp.ndarray:
    """Policy-optimized IK combining pose matching with joint continuity
    and manipulability maximization.

    Produces configurations that are smoother across consecutive calls and
    favor high-manipulability arm postures, yielding trajectories that are
    easier for a downstream policy to learn.

    Args:
        robot: PyRoKi Robot.
        target_link_name: Name of the end-effector link.
        target_wxyz: (4,) Target orientation as WXYZ quaternion.
        target_position: (3,) Target position.
        prev_cfg: Previous joint configuration for continuity.
        initial_cfg: Optional initial guess for the solver.
        continuity_weight: Weight for staying close to prev_cfg (smoothness).
        manipulability_weight: Weight for maximizing Yoshikawa manipulability.

    Returns:
        cfg: (num_actuated_joints,) Optimized joint configuration.
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_index = robot.links.names.index(target_link_name)
    init = None if initial_cfg is None else jnp.array(initial_cfg)
    cfg = _solve_ik_policy_optimized_jax(
        robot,
        jnp.array(target_link_index, dtype=jnp.int32),
        jnp.array(target_wxyz),
        jnp.array(target_position),
        jnp.array(prev_cfg),
        init,
        jnp.array(continuity_weight),
        jnp.array(manipulability_weight),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return onp.array(cfg)


@jdc.jit
def _solve_ik_policy_optimized_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    prev_cfg: jax.Array,
    initial_cfg: jax.Array | None,
    continuity_weight: jax.Array,
    manipulability_weight: jax.Array,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)

    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )

    factors = [
        # Primary: reach the target pose
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            target_pose,
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        # Joint limits (hard boundary)
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=100.0,
        ),
        # Continuity: bias toward previous configuration for smoothness
        pk.costs.rest_cost(
            joint_var,
            prev_cfg,
            jnp.full(robot.joints.num_actuated_joints, continuity_weight),
        ),
        # Manipulability: favor high-manipulability configurations
        pk.costs.manipulability_cost(
            robot,
            joint_var,
            target_link_indices=target_link_index,
            weight=manipulability_weight,
        ),
    ]

    initial_vals = None
    if initial_cfg is not None:
        initial_vals = jaxls.VarValues.make(
            (joint_var.with_value(initial_cfg),)
        )

    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            initial_vals=initial_vals,
        )
    )
    return sol[joint_var]
