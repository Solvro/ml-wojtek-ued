"""The deterministic actor as an ONNX graph: raw `obs` in, `action` and
`joint_targets` out. The observation normalizer is part of the graph."""

from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from config.schema import Activation, NetworkConfig
from export.contract import Contract

OPSET = 17
IR_VERSION = 8  # the IR that opset 17 shipped with; old runtimes load it
INPUT = "obs"
OUTPUTS = ("action", "joint_targets")


class Graph:
    """Accumulates nodes and constant tensors; every op returns its output."""

    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def const(self, name: str, value: Any, dtype: Any = np.float32) -> str:
        array = np.asarray(value, dtype=dtype)
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def op(
        self, op_type: str, inputs: list[str], name: str, **attrs: Any
    ) -> str:
        self.nodes.append(
            helper.make_node(op_type, inputs, [name], name=name, **attrs)
        )
        return name


def policy_layers(
    policy_params: Any, network: NetworkConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(kernel, bias) per layer of the flax MLP, input to output."""
    layers = policy_params["params"]
    names = [f"hidden_{i}" for i in range(len(network.policy_hidden) + 1)]
    if sorted(layers) != sorted(names):
        raise ValueError(
            f"checkpoint policy layers {sorted(layers)} do not match "
            f"network.policy_hidden={list(network.policy_hidden)} "
            f"(expected {names}); pass the network config the run trained with"
        )
    return [
        (
            np.asarray(layers[n]["kernel"], np.float32),
            np.asarray(layers[n]["bias"], np.float32),
        )
        for n in names
    ]


def check_obs_dim(normalizer: Any, layers: list[Any], obs_dim: int) -> None:
    mean = np.asarray(normalizer.mean["state"])
    sizes = {
        "normalizer": mean.shape[-1],
        "first policy layer": layers[0][0].shape[0],
    }
    for what, size in sizes.items():
        if size != obs_dim:
            raise ValueError(
                f"the checkpoint's {what} takes {size} actor observations, "
                f"the config's env produces {obs_dim}. The checkpoint was "
                "trained with a different robot or observation layout: pass "
                "the config of that run"
            )


def activation(g: Graph, kind: Activation, x: str, i: int) -> str:
    name = f"act_{i}"
    if kind == Activation.elu:
        return g.op("Elu", [x], name, alpha=1.0)
    if kind == Activation.relu:
        return g.op("Relu", [x], name)
    if kind == Activation.tanh:
        return g.op("Tanh", [x], name)
    sigmoid = g.op("Sigmoid", [x], f"{name}_sigmoid")
    return g.op("Mul", [x, sigmoid], name)


def actor(
    g: Graph, params: Any, network: NetworkConfig, contract: Contract
) -> str:
    normalizer, policy_params = params[0], params[1]
    layers = policy_layers(policy_params, network)
    check_obs_dim(normalizer, layers, contract.obs_dim)
    mean = np.asarray(normalizer.mean["state"], np.float32)
    std = np.asarray(normalizer.std["state"], np.float32)
    x = g.op("Sub", [INPUT, g.const("obs_mean", mean[None])], "obs_centered")
    x = g.op("Div", [x, g.const("obs_std", std[None])], "obs_normalized")
    for i, (kernel, bias) in enumerate(layers[:-1]):
        w, b = g.const(f"w{i}", kernel), g.const(f"b{i}", bias)
        x = activation(
            g, network.activation, g.op("Gemm", [x, w, b], f"fc{i}"), i
        )
    kernel, bias = layers[-1]
    n = contract.action_dim
    # the head is [mean, scale]; the deterministic action is tanh(mean)
    w = g.const("w_out", kernel[:, :n])
    b = g.const("b_out", bias[:n])
    mean_head = g.op("Gemm", [x, w, b], "fc_out")
    return g.op("Tanh", [mean_head], OUTPUTS[0])


def clamp(
    g: Graph, x: str, lo: np.ndarray, hi: np.ndarray, name: str, out: str = ""
) -> str:
    """Per-joint bounds; an infinite bound becomes float32's extreme."""
    f32 = np.finfo(np.float32)
    lo = np.clip(lo, f32.min, f32.max)
    hi = np.clip(hi, f32.min, f32.max)
    x = g.op("Max", [x, g.const(f"{name}_lo", lo[None])], f"{name}_max")
    return g.op(
        "Min", [x, g.const(f"{name}_hi", hi[None])], out or f"{name}_min"
    )


def height_dsecond(g: Graph, contract: Contract) -> str:
    """np.interp of the commanded height onto the stance knots: the first
    knot value plus clipped ramps, flat beyond both ends."""
    stance = contract.stance
    assert stance is not None
    xs = np.asarray(stance.knots_height, np.float64)
    ys = np.asarray(stance.knots_dsecond, np.float64)
    width = np.diff(xs)
    slope = np.diff(ys) / width
    i = stance.height_index
    height = g.op(
        "Slice",
        [
            INPUT,
            g.const("h_start", [i], np.int64),
            g.const("h_stop", [i + 1], np.int64),
            g.const("h_axis", [1], np.int64),
        ],
        "cmd_height",
    )
    over = g.op("Sub", [height, g.const("knot_x", xs[:-1][None])], "h_over")
    ramp = g.op(
        "Min",
        [g.op("Relu", [over], "h_ramp"), g.const("knot_w", width[None])],
        "h_span",
    )
    rise = g.op("Mul", [ramp, g.const("knot_slope", slope[None])], "h_rise")
    total = g.op(
        "ReduceSum",
        [rise, g.const("sum_axis", [1], np.int64)],
        "h_total",
        keepdims=1,
    )
    return g.op("Add", [total, g.const("knot_y0", [[ys[0]]])], "dsecond")


def stance_pose(g: Graph, contract: Contract) -> str:
    stance = contract.stance
    assert stance is not None
    dsecond = height_dsecond(g, contract)
    offset = g.op(
        "Mul",
        [dsecond, g.const("leg_offsets", np.asarray(stance.offsets)[None])],
        "stance_offset",
    )
    pose = g.op(
        "Add",
        [offset, g.const("home", np.asarray(stance.home)[None])],
        "stance_raw",
    )
    return clamp(
        g,
        pose,
        np.asarray(stance.stance_min),
        np.asarray(stance.stance_max),
        "stance",
    )


def joint_targets(g: Graph, action: str, contract: Contract) -> str:
    joints = contract.joints
    scale = np.asarray([j.action_scale for j in joints])
    lo = np.asarray([j.target_min for j in joints])
    hi = np.asarray([j.target_max for j in joints])
    if contract.stance is None:
        origin = g.const(
            "default_pose", np.asarray([j.default for j in joints])[None]
        )
    else:
        origin = stance_pose(g, contract)
    moved = g.op(
        "Mul", [action, g.const("action_scale", scale[None])], "action_offset"
    )
    raw = g.op("Add", [origin, moved], "targets_raw")
    return clamp(g, raw, lo, hi, "targets", OUTPUTS[1])


def build_model(
    params: Any, network: NetworkConfig, contract: Contract
) -> onnx.ModelProto:
    g = Graph()
    action = actor(g, params, network, contract)
    joint_targets(g, action, contract)
    n = contract.action_dim
    graph = helper.make_graph(
        g.nodes,
        "policy",
        [
            helper.make_tensor_value_info(
                INPUT, TensorProto.FLOAT, ["batch", contract.obs_dim]
            )
        ],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", n])
            for name in OUTPUTS
        ],
        initializer=g.initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="wojtek-ued",
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model, full_check=True)
    return model
