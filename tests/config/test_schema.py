from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException, MissingConfigException
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError, MissingMandatoryValue

from config.hfield_config import HFieldConfig
from config.schema import (
    BC,
    CONF_DIR,
    Activation,
    Backend,
    Curriculum,
    LevelSpaceKind,
    Mutation,
    Robot,
    Score,
    TrainConfig,
    as_dict,
    check_group_options,
    to_dataclass,
)

Compose = Callable[..., Any]

REQUIRED = {
    "train": ("mode=dr", "name=t"),
    "profile": (),
    "eval": ("checkpoint=c.pkl", "name=t", "out_dir=o"),
    "sim2sim": ("checkpoint=c.pkl", "name=t", "out_dir=o"),
    "render": ("checkpoint=c.pkl", "levels=[0.75:3]"),
    "export": ("checkpoint=c.pkl", "mode=dr", "name=t"),
    "arena": (),
    "check_hfield": (),
    "paper_tables": (),
    "paper_figures": (),
    "domination": ("eval_dir=e",),
}
GROUPS = {
    "train": ("robot", "mode", "level_space", "arena", "network"),
    "profile": ("robot", "mode", "level_space", "arena", "network"),
    "eval": ("robot", "level_space", "arena", "network"),
    "sim2sim": ("robot", "arena", "network"),
    "render": ("robot", "level_space", "arena", "network"),
    "export": ("robot", "mode", "level_space", "arena", "network"),
    "arena": ("arena",),
    "check_hfield": ("arena",),
    "paper_figures": ("arena",),
}
OPTIONS = {
    "robot": ("wojtek", "go1", "barkour", "spot", "go2"),
    "mode": (
        "dr",
        "dr_easy",
        "dr_mixed",
        "terrain_cl",
        "accel",
        "accel_grow",
        "accel_sfl",
        "accel_lp",
        "plr",
        "sfl",
        "paired",
        "paired_uni",
        "paired_bi",
        "paired_evo",
    ),
    "level_space": ("grid", "param"),
    "arena": ("benchmark",),
    "network": ("default",),
}
BENCHMARK = HFieldConfig(
    amplitude=0.3, coarse_n=13, pad_radius=0.55, pad_taper=0.2
)


@pytest.mark.parametrize("name", REQUIRED)
def test_every_primary_composes_with_its_defaults(
    compose_cfg: Compose, name: str
) -> None:
    compose_cfg(name, *REQUIRED[name])


@pytest.mark.parametrize(
    ("name", "group", "option"),
    [
        (name, group, option)
        for name, groups in GROUPS.items()
        for group in groups
        for option in OPTIONS[group]
    ],
)
def test_every_group_option_composes(
    compose_cfg: Compose, name: str, group: str, option: str
) -> None:
    overrides = [o for o in REQUIRED[name] if not o.startswith(f"{group}=")]
    compose_cfg(name, *overrides, f"{group}={option}")


def test_train_defaults(compose_cfg: Compose) -> None:
    cfg = compose_cfg("train", "mode=dr", "name=t")
    assert cfg.robot.name == Robot.wojtek
    assert cfg.level_space.kind == LevelSpaceKind.param
    assert (cfg.level_space.n_edits, cfg.level_space.edit_scale) == (2, 0.1)
    assert cfg.arena == BENCHMARK
    assert cfg.backend == Backend.auto
    assert (cfg.seed, cfg.num_timesteps) == (0, 300_000_000)
    assert (cfg.num_lanes, cfg.envs_per_lane, cfg.level_steps) == (256, 32, 300)
    assert cfg.checkpoint_dir is None
    assert cfg.tags == []


@pytest.mark.parametrize(
    ("mode", "curriculum", "mutation", "score"),
    [
        ("dr", Curriculum.dr, Mutation.local, Score.pvl),
        ("dr_easy", Curriculum.dr_easy, Mutation.local, Score.pvl),
        ("dr_mixed", Curriculum.dr_mixed, Mutation.local, Score.pvl),
        ("accel", Curriculum.accel, Mutation.local, Score.pvl),
        ("accel_grow", Curriculum.accel, Mutation.grow, Score.pvl),
        ("accel_sfl", Curriculum.accel, Mutation.grow, Score.learnability),
        ("accel_lp", Curriculum.accel, Mutation.local, Score.lp),
        ("terrain_cl", Curriculum.terrain_cl, Mutation.local, Score.pvl),
        ("plr", Curriculum.plr, Mutation.local, Score.pvl),
        ("sfl", Curriculum.plr, Mutation.local, Score.learnability),
        ("paired", Curriculum.paired, Mutation.local, Score.pvl),
    ],
)
def test_mode_sets_its_curriculum_switches(
    compose_cfg: Compose,
    mode: str,
    curriculum: Curriculum,
    mutation: Mutation,
    score: Score,
) -> None:
    cfg = compose_cfg("train", f"mode={mode}", "name=t")
    assert cfg.mode.curriculum == curriculum
    assert cfg.mode.mutation == mutation
    assert cfg.mode.score == score
    assert cfg.mode.generator_scale == 0.1
    assert cfg.mode.buffer_size == 4000


@pytest.mark.parametrize(
    ("mode", "curriculum", "bc"),
    [
        ("paired", Curriculum.paired, BC.none),
        ("paired_uni", Curriculum.paired, BC.uni),
        ("paired_bi", Curriculum.paired, BC.bi),
        ("paired_evo", Curriculum.paired_evo, BC.none),
    ],
)
def test_paired_modes_set_their_distillation(
    compose_cfg: Compose, mode: str, curriculum: Curriculum, bc: BC
) -> None:
    cfg = compose_cfg("train", f"mode={mode}", "name=t")
    assert (cfg.mode.curriculum, cfg.mode.bc) == (curriculum, bc)
    assert (cfg.mode.bc_coef, cfg.mode.bc_every) == (0.5, 1)


def test_bc_is_a_field_on_paired_evo(compose_cfg: Compose) -> None:
    cfg = compose_cfg("train", "mode=paired_evo", "name=t", "mode.bc=bi")
    assert (cfg.mode.curriculum, cfg.mode.bc) == (Curriculum.paired_evo, BC.bi)


def test_student_entropy_default_is_brax_recipe(compose_cfg: Compose) -> None:
    for mode in OPTIONS["mode"]:
        cfg = compose_cfg("train", f"mode={mode}", "name=t")
        assert cfg.mode.student_entropy_coef == 0.01


@pytest.mark.parametrize(
    ("space", "n_edits", "edit_scale"), [("grid", 8, 0.15), ("param", 2, 0.1)]
)
def test_level_space_sets_its_edits(
    compose_cfg: Compose, space: str, n_edits: int, edit_scale: float
) -> None:
    cfg = compose_cfg("train", "mode=accel", "name=t", f"level_space={space}")
    assert cfg.level_space.kind == LevelSpaceKind(space)
    assert (cfg.level_space.n_edits, cfg.level_space.edit_scale) == (
        n_edits,
        edit_scale,
    )


def test_generator_scale_suffix_is_an_override(compose_cfg: Compose) -> None:
    cfg = compose_cfg(
        "train", "mode=accel_grow", "name=t", "mode.generator_scale=0.2"
    )
    assert cfg.mode.generator_scale == 0.2
    assert cfg.mode.mutation == Mutation.grow


def test_arena_builder_and_training_read_the_same_arena(
    compose_cfg: Compose,
) -> None:
    built = compose_cfg("arena").arena
    for name in ("train", "eval", "sim2sim", "render", "check_hfield"):
        assert compose_cfg(name, *REQUIRED[name]).arena == built == BENCHMARK
    tiles = compose_cfg("arena", "arena.tiles=8").arena
    assert (
        compose_cfg("train", "mode=dr", "name=t", "arena.tiles=8").arena
        == tiles
    )


def test_profile_is_a_param_dr_run(compose_cfg: Compose) -> None:
    cfg = compose_cfg("profile", "backend=warp", "robot=go1")
    assert cfg.mode.curriculum == Curriculum.dr
    assert cfg.level_space.kind == LevelSpaceKind.param
    assert (cfg.level_space.n_edits, cfg.level_space.edit_scale) == (2, 0.1)
    assert cfg.backend == Backend.warp
    assert cfg.robot.name == Robot.go1
    assert cfg.name == "profile"


def test_per_entry_point_defaults(compose_cfg: Compose) -> None:
    assert compose_cfg("eval", *REQUIRED["eval"]).level_space.kind == "param"
    assert (
        compose_cfg("render", *REQUIRED["render"]).level_space.kind == "param"
    )
    assert compose_cfg("eval", *REQUIRED["eval"]).alphas == [0.1, 0.25]
    assert compose_cfg("render", *REQUIRED["render"]).command == [0.3, 0.0, 0.2]


def test_lists_from_the_command_line(compose_cfg: Compose) -> None:
    cfg = compose_cfg("train", "mode=dr", "name=t", "tags=[a,b]")
    assert cfg.tags == ["a", "b"]
    render = compose_cfg("render", "checkpoint=c", "levels=[0.5:1,1.0:17]")
    assert render.levels == ["0.5:1", "1.0:17"]


@pytest.mark.parametrize(
    "name", ["paper_tables", "paper_figures", "domination"]
)
def test_paper_prefixes_default_to_the_slurm_run_names(
    compose_cfg: Compose, name: str
) -> None:
    cfg = compose_cfg(name, *REQUIRED[name])
    assert cfg.prefixes == {robot: f"ued_{robot}" for robot in Robot}


def test_paper_prefixes_per_robot(compose_cfg: Compose) -> None:
    cfg = compose_cfg("paper_tables", "prefixes.go1=x", "prefixes.spot=null")
    assert cfg.prefixes == {
        Robot.wojtek: "ued_wojtek",
        Robot.go1: "x",
        Robot.barkour: "ued_barkour",
        Robot.spot: None,
        Robot.go2: "ued_go2",
    }


def test_as_dict_writes_enums_as_strings(compose_cfg: Compose) -> None:
    out = as_dict(compose_cfg("train", "mode=accel_sfl", "name=t"))
    assert out["mode"]["score"] == "learnability"
    assert type(out["robot"]["name"]) is str


@pytest.mark.parametrize(
    "overrides",
    [
        ("mode=dr", "nmae=t"),
        ("mode=dr", "name=t", "mode.scor=pvl"),
        ("mode=dr", "name=t", "arena.amplitdue=0.3"),
        ("mode=dr", "name=t", "seed=abc"),
        ("mode=dr", "name=t", "backend=cuda"),
        ("mode=dr", "name=t", "mode.score=learnabilty"),
        ("mode=paired", "name=t", "mode.bc=unibc"),
        ("mode=dr", "name=t", "prefixes.go1=x"),
    ],
)
def test_typos_fail_at_startup(
    compose_cfg: Compose, overrides: tuple[str]
) -> None:
    with pytest.raises(ConfigCompositionException):
        compose_cfg("train", *overrides)


def test_unknown_group_option_fails(compose_cfg: Compose) -> None:
    with pytest.raises(MissingConfigException, match="anymal"):
        compose_cfg("train", "mode=dr", "name=t", "robot=anymal")


def test_mode_is_required(compose_cfg: Compose) -> None:
    with pytest.raises(ConfigCompositionException, match="mode"):
        compose_cfg("train", "name=t")


@pytest.mark.parametrize(
    ("name", "missing"),
    [("train", "name"), ("eval", "checkpoint"), ("render", "levels")],
)
def test_required_fields_fail_when_unset(
    compose_cfg: Compose, name: str, missing: str
) -> None:
    overrides = [o for o in REQUIRED[name] if not o.startswith(f"{missing}=")]
    with pytest.raises(MissingMandatoryValue, match=missing):
        compose_cfg(name, *overrides)


def test_run_dir_without_name_says_what_to_pass() -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose("train", overrides=["mode=dr"], return_hydra_config=True)
        with pytest.raises(InterpolationResolutionError, match="pass name="):
            _ = cfg.hydra.run.dir


@pytest.mark.parametrize("key", ["n_edits", "edit_scale"])
def test_edit_strength_is_only_a_level_space_key(
    compose_cfg: Compose, key: str
) -> None:
    with pytest.raises(ConfigCompositionException, match=key):
        compose_cfg("train", "mode=dr", "name=t", f"arena.{key}=3")


@pytest.mark.parametrize(
    ("space", "mode", "n_edits"),
    [
        ("grid", "accel", 8),
        ("grid", "accel_grow", 16),
        ("grid", "accel_sfl", 16),
        ("param", "accel", 2),
        ("param", "accel_grow", 2),
        ("param", "accel_sfl", 2),
    ],
)
def test_grow_variants_edit_more_grid_cells(
    compose_cfg: Compose, space: str, mode: str, n_edits: int
) -> None:
    cfg = compose_cfg("train", f"mode={mode}", "name=t", f"level_space={space}")
    assert cfg.level_space.n_edits == n_edits


def test_n_edits_override_wins_over_the_mutation_default(
    compose_cfg: Compose,
) -> None:
    cfg = compose_cfg(
        "train", "mode=accel_grow", "name=t", "level_space.n_edits=4"
    )
    assert cfg.level_space.n_edits == 4


def test_run_dir_records_the_config_beside_the_run() -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            "train", overrides=["mode=dr", "name=x"], return_hydra_config=True
        )
    assert cfg.hydra.run.dir == "artifacts/hydra/x"
    assert cfg.hydra.job.chdir is False


# mujoco_playground's locomotion recipe, the constants training/ppo.py held
OLD_PPO = {
    "unroll_length": 20,
    "num_minibatches": 32,
    "num_updates_per_batch": 4,
    "discounting": 0.97,
    "gae_lambda": 0.95,
    "learning_rate": 3e-4,
    "clipping_epsilon": 0.2,
    "max_grad_norm": 1.0,
}


def test_ppo_defaults_are_the_old_constants(compose_cfg: Compose) -> None:
    cfg = compose_cfg("train", "mode=dr", "name=t")
    assert as_dict(cfg.ppo) == OLD_PPO


@pytest.mark.parametrize("name", ["train", "eval", "sim2sim", "render"])
def test_network_defaults_are_the_old_constants(
    compose_cfg: Compose, name: str
) -> None:
    net = compose_cfg(name, *REQUIRED[name]).network
    assert net.policy_hidden == [512, 256, 128]
    assert net.value_hidden == [512, 256, 128]
    assert net.activation == Activation.elu


def test_ppo_and_network_are_overridable(compose_cfg: Compose) -> None:
    cfg = compose_cfg(
        "train",
        "mode=dr",
        "name=t",
        "ppo.learning_rate=1e-3",
        "network.policy_hidden=[64,64]",
        "network.activation=swish",
    )
    assert cfg.ppo.learning_rate == 1e-3
    assert cfg.network.policy_hidden == [64, 64]
    assert cfg.network.activation == Activation.swish


@pytest.mark.parametrize(
    "overrides",
    [("ppo.learning=1e-3",), ("network.activation=gelu",)],
)
def test_ppo_and_network_typos_fail_at_startup(
    compose_cfg: Compose, overrides: tuple[str]
) -> None:
    with pytest.raises(ConfigCompositionException):
        compose_cfg("train", "mode=dr", "name=t", *overrides)


# --- conf/policy/template.yaml ---------------------------------------------


def test_template_composes_without_a_mode(compose_cfg: Compose) -> None:
    cfg = compose_cfg("train", "policy=template")
    assert cfg.name == "my_policy"
    assert cfg.robot.name == Robot.wojtek
    assert cfg.mode.curriculum == Curriculum.accel


def test_template_matches_the_benchmark_accel_config(
    compose_cfg: Compose,
) -> None:
    template = as_dict(compose_cfg("train", "policy=template"))
    bench = as_dict(compose_cfg("train", "mode=accel", "name=my_policy"))
    # the run bookkeeping the template sets for local policy runs
    for key in ("project", "tags", "checkpoint_dir"):
        template.pop(key)
        bench.pop(key)
    assert template == bench


def leaves(node: Any, prefix: str = "") -> set[str]:
    if not isinstance(node, dict):
        return {prefix}
    return {
        leaf
        for key, value in node.items()
        for leaf in leaves(value, f"{prefix}{key}.")
    }


def test_template_lists_every_value_of_the_train_config() -> None:
    raw = OmegaConf.to_container(
        OmegaConf.load(Path(CONF_DIR) / "policy/template.yaml")
    )
    assert isinstance(raw, dict)
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        composed = compose("train", overrides=["mode=accel", "name=t"])
    wanted = leaves(as_dict(to_dataclass(composed, TrainConfig)))
    # `defaults` is the template's choice list, the rest are values
    listed = leaves({k: v for k, v in raw.items() if k != "defaults"})
    assert wanted - listed == set()


def template_choices(**changed: str) -> dict[str, str]:
    return {
        "robot": "wojtek",
        "mode": "accel",
        "level_space": "param",
        "arena": "benchmark",
        "network": "default",
    } | changed


def test_template_matches_its_group_options(compose_cfg: Compose) -> None:
    cfg = OmegaConf.structured(compose_cfg("train", "policy=template"))
    check_group_options(cfg, template_choices())


@pytest.mark.parametrize(
    ("group", "option", "key"),
    [
        ("robot", "go1", "robot.name"),
        ("mode", "sfl", "mode.curriculum"),
        ("level_space", "grid", "level_space.kind"),
    ],
)
def test_group_choice_the_template_overrides_is_refused(
    compose_cfg: Compose, group: str, option: str, key: str
) -> None:
    cfg = OmegaConf.structured(
        compose_cfg("train", "policy=template", f"{group}={option}")
    )
    with pytest.raises(ValueError, match=f"{group}={option} sets {key}"):
        check_group_options(cfg, template_choices(**{group: option}))


def test_value_override_of_the_template_passes_the_group_check(
    compose_cfg: Compose,
) -> None:
    cfg = OmegaConf.structured(
        compose_cfg("train", "policy=template", "mode.score=learnability")
    )
    check_group_options(cfg, template_choices())


def test_cli_overrides_win_over_the_template(compose_cfg: Compose) -> None:
    cfg = compose_cfg(
        "train", "policy=template", "seed=7", "ppo.max_grad_norm=0.5"
    )
    assert (cfg.seed, cfg.ppo.max_grad_norm) == (7, 0.5)


def test_unknown_policy_fails(compose_cfg: Compose) -> None:
    with pytest.raises(MissingConfigException, match="nothing"):
        compose_cfg("train", "policy=nothing")
