"""The README that ships beside policy.onnx: the deployment contract,
rendered from a Contract and the run's config."""

import json
import math

from config.schema import TrainConfig
from export.contract import (
    GAIT_STAND_SPEED,
    YAW_SPEED_WEIGHT,
    Contract,
    index_ranges,
)
from export.onnx_graph import INPUT, OPSET, OUTPUTS
from export.verify import Verification
from robots.scale import level_steps, min_travel
from training.summary import TrainingSummary

SECTIONS_SEP = "\n\n"
SENSOR_PLACEHOLDER = "..."
READ = {
    "gyro": "robot.imu_gyro()",
    "gravity": "gravity_from_quaternion(robot.imu_quaternion())",
    "joint_pos": "robot.joint_positions() - DEFAULT_POSE",
    "joint_vel": "robot.joint_velocities()",
    "last_act": "last_action",
    "command": "command",
    "phase": "gait_phase_features(phase, command)",
    "linvel": "estimator.body_velocity()",
    "gyro_z": "0.25 * robot.imu_gyro()[2]",
    "qpos_error_history": "qpos_error_history",
    "feet_pos": "robot.foot_positions()",
}


def num(x: float) -> str:
    if math.isinf(x):
        return "-inf" if x < 0 else "inf"
    return f"{x:.6g}"


def array(values: list[float] | tuple[float, ...]) -> str:
    return "[" + ", ".join(num(v) for v in values) + "]"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def header_section(
    name: str, cfg: TrainConfig, contract: Contract, ver: Verification
) -> str:
    rate = 1.0 / contract.ctrl_dt
    rows = [
        ["robot", f"`{contract.robot}`"],
        ["training", f"`mode={cfg.mode.curriculum}`, seed {cfg.seed}"],
        ["input", f"`{INPUT}` float32 `[batch, {contract.obs_dim}]`"],
        [
            "outputs",
            ", ".join(
                f"`{o}` float32 `[batch, {contract.action_dim}]`"
                for o in OUTPUTS
            ),
        ],
        [
            "control period",
            f"`ctrl_dt = {num(contract.ctrl_dt)} s` ({num(rate)} Hz)",
        ],
        ["ONNX opset", str(OPSET)],
        [
            "files",
            "`policy.onnx`, `config.yaml` (full training config), `README.md`",
        ],
    ]
    return (
        f"# {name}\n\n"
        f"Deterministic `{contract.robot}` locomotion policy exported to ONNX: "
        "the actor MLP with the observation normalizer folded in, no "
        "sampling (`action = tanh(mean)`).\n\n" + table(["", ""], rows)
    )


def inputs_section(contract: Contract) -> str:
    rows = []
    for component, start, stop in index_ranges(contract):
        noise = "-" if component.noise is None else num(component.noise)
        rows.append(
            [
                f"`{component.name}`",
                f"{start}:{stop}",
                str(component.size),
                component.units,
                component.source,
                noise,
            ]
        )
    head = [
        "component",
        "indices",
        "size",
        "units",
        "read on the robot",
        "train noise (+-)",
    ]
    text = (
        "## Inputs\n\n"
        f"`{INPUT}` is the raw observation, concatenated in the order below "
        f"({contract.frame_dim} values"
        + (
            f" per frame, {contract.frames} frames"
            if contract.frames > 1
            else ""
        )
        + f", {contract.obs_dim} total). Do not normalize it: the "
        "training-time running mean / std are inside the graph.\n\n"
        + table(head, rows)
    )
    if contract.frames > 1:
        text += (
            f"\n\nThe table is one frame, newest first. Older frames follow "
            f"at offsets of {contract.frame_dim}: the full vector is "
            f"`[frame_t, frame_t-1, ..., frame_t-{contract.frames - 1}]`; "
            "each step prepend the new frame and drop the oldest. "
            "Frames before the start are zeros."
        )
    return text


def outputs_section(contract: Contract) -> str:
    n = contract.action_dim
    text = (
        "## Outputs\n\n"
        f"- `action` `[batch, {n}]`: the policy action in [-1, 1]; feed it "
        "back as `last_act` on the next step.\n"
        f"- `joint_targets` `[batch, {n}]`: rad, the position setpoints for "
        "the joint PD controllers (joint order below)."
    )
    return text


def joints_section(contract: Contract) -> str:
    rows = []
    for i, j in enumerate(contract.joints):
        torque = "-" if j.torque_limit is None else num(j.torque_limit)
        rows.append(
            [
                str(i),
                f"`{j.name}`",
                num(j.default),
                num(j.target_min),
                num(j.target_max),
                num(j.kp),
                num(j.kd),
                torque,
                num(j.action_scale),
            ]
        )
    head = [
        "#", "joint", "default (rad)", "target min", "target max",
        "kp", "kd", "torque limit (N m)", "action_scale",
    ]  # fmt: skip
    return (
        "## Joints and gains\n\n"
        "Joint order is the order of `joint_pos`, `joint_vel`, `last_act`, "
        "`action` and `joint_targets`. `kp` / `kd` are the PD gains of the "
        "simulated joints (`kd` = actuator velocity gain + joint damping); "
        "the real controller must use the same.\n\n" + table(head, rows)
    )


def stance_text(contract: Contract) -> str:
    stance = contract.stance
    assert stance is not None
    knots = table(
        ["height (m)", "dsecond (rad)"],
        [
            [num(h), num(d)]
            for h, d in zip(stance.knots_height, stance.knots_dsecond)
        ],
    )
    return (
        "- `stance(height) = clip(default + OFFSETS * dsecond(height), "
        "stance_min, stance_max)` with `OFFSETS = "
        f"{array(stance.offsets)}` and `dsecond` the linear interpolation "
        "(flat beyond the ends) of the table below.\n"
        f"  `height` is `command[3]` (obs index {stance.height_index}).\n"
        f"  `stance_min = {array(stance.stance_min)}`,\n"
        f"  `stance_max = {array(stance.stance_max)}`.\n\n" + knots + "\n\n"
    )


def control_section(contract: Contract) -> str:
    text = (
        "## Joint targets\n\n"
        f"- `ctrl_dt = {num(contract.ctrl_dt)} s`: run the policy and send "
        "targets at this period.\n"
    )
    if contract.stance is None:
        text += (
            "- `joint_targets = clip(default + action_scale * action, "
            "target_min, target_max)`.\n"
        )
    else:
        text += (
            "- `joint_targets = clip(stance(height) + action_scale * "
            "action, target_min, target_max)`; a zero action stands at the "
            "commanded height.\n"
        )
        text += stance_text(contract)
    if contract.action_filter:
        a = contract.action_filter
        text += (
            f"- The env low-passes the action before the targets: "
            f"`filt = {num(a)} * filt_prev + {num(1 - a)} * action`, and "
            "applies `action_scale * filt`. The graph's `joint_targets` "
            "uses `action` directly, so the control loop corrects it "
            "(example below); this matches the env except where a target "
            "sits on `target_min` / `target_max`.\n"
        )
    return text.rstrip()


def commands_section(contract: Contract) -> str:
    c = contract.command
    rows = [
        [f"`{n}`", u, num(lo), num(hi)]
        for n, u, lo, hi in zip(c.names, c.units, c.low, c.high)
    ]
    head = ["command", "units", "low", "high"]
    notes = "\n".join(n if n.startswith("  ") else f"- {n}" for n in c.notes)
    return (
        "## Commands\n\n"
        "The command components are part of the observation; the policy was "
        "trained on commands in these ranges (base frame, x forward, y "
        "left, yaw counter-clockwise). Stay inside them.\n\n"
        + table(head, rows)
        + "\n\n"
        + notes
    )


def gait_section(contract: Contract) -> str:
    g = contract.gait_clock
    if g is None:
        return ""
    return (
        "## Gait clock\n\n"
        "`phase` is not a sensor: the robot runs the same clock as the "
        "env, starting at `phase = 0` and ticking once per control step, "
        "after the step, with the command that was active during it.\n\n"
        "```python\n"
        f"WALK_PHASE = np.array({array(g.walk_phase)})\n"
        f"TROT_PHASE = np.array({array(g.trot_phase)})\n"
        f"FREQ = {array(g.freq)}  # Hz at the slowest / fastest command\n"
        f"TROT_BAND = {array(g.trot_band)}  # m/s, walk -> trot blend\n"
        f"MAX_SPEED = {num(g.max_speed)}  # m/s\n"
        f"CTRL_DT = {num(g.dt)}\n\n\n"
        "def wrap(angle):\n"
        "    return np.fmod(angle + np.pi, 2 * np.pi) - np.pi\n\n\n"
        "def command_speed(command):\n"
        "    return np.hypot(command[0], command[1]) + "
        f"{num(YAW_SPEED_WEIGHT)} * abs(command[2])\n\n\n"
        "def advance_phase(phase, command):\n"
        "    speed = command_speed(command)\n"
        f"    if speed <= {num(GAIT_STAND_SPEED)}:  # standing: clock frozen\n"
        "        return phase\n"
        "    fast = min(speed / MAX_SPEED, 1.0)\n"
        "    freq = FREQ[0] + (FREQ[1] - FREQ[0]) * fast\n"
        "    return wrap(phase + 2 * np.pi * CTRL_DT * freq)\n\n\n"
        "def gait_phase_features(phase, command):\n"
        "    blend = np.clip(\n"
        "        (command_speed(command) - TROT_BAND[0])\n"
        "        / (TROT_BAND[1] - TROT_BAND[0]),\n"
        "        0.0,\n"
        "        1.0,\n"
        "    )\n"
        "    offsets = (1 - blend) * WALK_PHASE + blend * TROT_PHASE\n"
        "    legs = wrap(phase + offsets)\n"
        "    return np.concatenate([np.cos(legs), np.sin(legs)])\n"
        "```\n\n"
        "Per step: build the observation from "
        "`gait_phase_features(phase, command)`, send the targets, then "
        "`phase = advance_phase(phase, command)`."
    )


BODY_GRAVITY_ROW = (
    "[-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))]"
)
WORLD_UP_ROW = (
    "[2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]"
)


def gravity_helper(contract: Contract) -> str:
    gravity = next((c for c in contract.obs if c.name == "gravity"), None)
    if gravity is None:
        return ""
    row = WORLD_UP_ROW if "(0, 0, 1)" in gravity.source else BODY_GRAVITY_ROW
    return (
        "def gravity_from_quaternion(q):  # q = (w, x, y, z), IMU\n"
        "    w, x, y, z = q\n"
        f"    return np.array({row})\n\n"
    )


def observation_lines(contract: Contract) -> str:
    lines = []
    for component in contract.obs:
        expr = READ.get(component.name, SENSOR_PLACEHOLDER)
        note = f"{component.name} ({component.size}), {component.units}"
        lines.append(f"            {expr},  # {note}")
    return "\n".join(lines)


def usage_section(contract: Contract) -> str:
    n, dim = contract.action_dim, contract.frame_dim
    default = array([j.default for j in contract.joints])
    pre = [
        "import time\n\nimport numpy as np\nimport onnxruntime as ort\n",
        (
            "session = ort.InferenceSession(\n"
            '    "policy.onnx", providers=["CPUExecutionProvider"]\n)'
        ),
        f"CTRL_DT = {num(contract.ctrl_dt)}",
        f"DEFAULT_POSE = np.array({default}, np.float32)",
    ]
    if contract.action_filter:
        scale = array([j.action_scale for j in contract.joints])
        lo = array([j.target_min for j in contract.joints])
        hi = array([j.target_max for j in contract.joints])
        pre += [
            f"FILTER = {num(contract.action_filter)}",
            f"ACTION_SCALE = np.array({scale}, np.float32)",
            f"TARGET_MIN = np.array({lo}, np.float32)",
            f"TARGET_MAX = np.array({hi}, np.float32)",
        ]
    body = [
        f"last_action = np.zeros({n}, np.float32)",
    ]
    if contract.action_filter:
        body.append(f"filtered = np.zeros({n}, np.float32)")
    if contract.gait_clock is not None:
        body.append("phase = 0.0")
    if contract.frames > 1:
        body.append(f"history = np.zeros({contract.obs_dim}, np.float32)")
    loop = [
        "while True:",
        "    started = time.monotonic()",
        "    command = ...  # operator setpoint, see Commands",
    ]
    frame = (
        "    frame = np.concatenate(\n        [\n"
        + observation_lines(contract)
        + "\n        ]\n    ).astype(np.float32)"
    )
    loop.append(frame)
    if contract.frames > 1:
        loop.append(f"    history = np.concatenate([frame, history[:-{dim}]])")
        loop.append("    obs = history[None]")
    else:
        loop.append("    obs = frame[None]")
    outputs = json.dumps(list(OUTPUTS))
    loop.append("    action, joint_targets = session.run(")
    loop.append(f'        {outputs}, {{"{INPUT}": obs}}')
    loop.append("    )")
    loop.append("    action, joint_targets = action[0], joint_targets[0]")
    if contract.action_filter:
        loop += [
            "    filtered = FILTER * filtered + (1 - FILTER) * action",
            "    joint_targets = np.clip(",
            "        joint_targets + ACTION_SCALE * (filtered - action),",
            "        TARGET_MIN,",
            "        TARGET_MAX,",
            "    )",
        ]
    loop += [
        "    robot.send_position_targets(joint_targets)  # your PD interface",
        "    last_action = action",
        *(
            ["    phase = advance_phase(phase, command)"]
            if contract.gait_clock is not None
            else []
        ),
        "    time.sleep(max(0.0, CTRL_DT - (time.monotonic() - started)))",
    ]
    code = (
        pre[0]
        + "\n"
        + gravity_helper(contract)
        + "\n".join(pre[1:])
        + "\n\n"
        + "robot = ...  # your hardware interface\n"
        + "\n".join(body)
        + "\n\n"
        + "\n".join(loop)
    )
    return (
        "## Usage\n\n"
        "Control loop sketch (`robot` is your hardware interface; run "
        "this at the control period, with the same sensor conventions as "
        "the table above):\n\n"
        f"```python\n{code}\n```"
    )


def conditions_section(contract: Contract) -> str:
    items = "\n".join(f"- {c}" for c in contract.conditions)
    return (
        "## Training conditions\n\n"
        "What the policy saw in simulation; mismatches on the robot are "
        "the usual sim2real gap.\n\n" + items
    )


def metrics_section(
    summary: TrainingSummary | None, threshold: float, steps: int
) -> str:
    head = "## Training metrics\n\n"
    if summary is None:
        return head + (
            "Not recorded: no `<checkpoint>.train.yaml` was found beside "
            "the checkpoint."
        )
    rows = [
        ["reward per step", num(summary.reward_per_step)],
        [
            f"train success (no fall, >= {threshold:.2f} m in {steps} steps)",
            num(summary.train_success),
        ],
        ["falls per step", num(summary.falls_per_step)],
        ["forward velocity (m/s)", num(summary.forward_vel)],
        ["commanded vx (m/s)", num(summary.cmd_vx)],
        ["env steps", str(summary.env_steps)],
        ["wall time (h)", f"{summary.wall_time_s / 3600:.2f}"],
    ]
    return (
        head + f"Means over the last {summary.window_iterations} training "
        "iterations, stochastic policy on the training levels (not an "
        "evaluation).\n\n" + table(["metric", "value"], rows)
    )


def verification_section(ver: Verification) -> str:
    return (
        "## Verification\n\n"
        "At export, onnxruntime was compared with the JAX policy and the "
        "env's action-to-target mapping (both outputs):\n\n"
        f"- {ver.random_samples} random observations: max abs error "
        f"{ver.random_error:.2e}\n"
        f"- {ver.env_samples} observations from env rollouts: max abs "
        f"error {ver.env_error:.2e}\n"
        f"- tolerance {ver.tolerance:.0e}"
    )


def reproduce_section(name: str) -> str:
    return (
        "## Reproduce\n\n"
        "`config.yaml` is the full composed training config: every value "
        "this policy was trained with. Re-export a checkpoint with it:\n\n"
        "```bash\n"
        f"just export checkpoint=<{name}.pkl> "
        f"config=artifacts/export/{name}/config.yaml\n"
        "```"
    )


def render_readme(
    name: str,
    cfg: TrainConfig,
    contract: Contract,
    ver: Verification,
    summary: TrainingSummary | None,
) -> str:
    sections = [
        header_section(name, cfg, contract, ver),
        inputs_section(contract),
        outputs_section(contract),
        joints_section(contract),
        control_section(contract),
        commands_section(contract),
        gait_section(contract),
        usage_section(contract),
        conditions_section(contract),
        metrics_section(
            summary,
            min_travel(cfg.robot, cfg.min_travel),
            level_steps(cfg.robot, cfg.level_steps),
        ),
        verification_section(ver),
        reproduce_section(name),
    ]
    return SECTIONS_SEP.join(s for s in sections if s) + "\n"
