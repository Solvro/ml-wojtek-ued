from dataclasses import dataclass, field
from typing import Any, cast

import jax
import jax.numpy as jnp
from brax.training import types

from training.paired import Teacher
from training.student import Student


@dataclass
class FakeObsSize:
    actor: int = 6
    critic: int = 8


@dataclass
class FakeEnv:
    observation_size: FakeObsSize = field(default_factory=FakeObsSize)
    action_size: int = 3


def make_batch(
    env: FakeEnv, envs: int = 32, steps: int = 20
) -> types.Transition:
    keys = iter(jax.random.split(jax.random.key(1), 7))

    def normal(*shape: int) -> jax.Array:
        return jax.random.normal(next(keys), shape)

    obs = {
        "state": normal(envs, steps, env.observation_size.actor),
        "privileged_state": normal(envs, steps, env.observation_size.critic),
    }
    return types.Transition(
        observation=obs,  # ty: ignore[invalid-argument-type]
        action=normal(envs, steps, env.action_size),
        reward=normal(envs, steps),
        discount=jnp.ones((envs, steps)),
        next_observation=obs,  # ty: ignore[invalid-argument-type]
        extras={  # ty: ignore[invalid-argument-type]
            "policy_extras": {
                "log_prob": normal(envs, steps),
                "raw_action": normal(envs, steps, env.action_size),
                "distribution_params": normal(envs, steps, 2 * env.action_size),
            },
            "state_extras": {"truncation": jnp.zeros((envs, steps))},
        },
    )


def updated_policy(
    student: Student,
    batch: types.Transition,
    teacher: Teacher | None,
    bc_coef: float,
) -> Any:
    params, *_ = student.update(
        student.params,
        student.opt_state,
        student.normalizer,
        batch,
        jax.random.key(2),
        teacher=teacher,
        bc_coef=jnp.float32(bc_coef),
    )
    return params.policy


def max_abs_diff(a: Any, b: Any) -> float:
    diffs = jax.tree.map(lambda x, y: jnp.abs(x - y).max(), a, b)
    return float(max(jax.tree.leaves(diffs)))


def test_bc_coef_changes_the_student_update() -> None:
    env = FakeEnv()
    student = Student(cast(Any, env), jax.random.key(0), 1e-3)
    teacher = Student(cast(Any, env), jax.random.key(7), 1e-3)
    batch = make_batch(env)
    no_teacher = updated_policy(student, batch, None, 0.0)
    zero_coef = updated_policy(student, batch, teacher.as_teacher(), 0.0)
    distilled = updated_policy(student, batch, teacher.as_teacher(), 0.5)
    assert max_abs_diff(zero_coef, no_teacher) < 1e-5
    assert max_abs_diff(distilled, no_teacher) > 1e-3
