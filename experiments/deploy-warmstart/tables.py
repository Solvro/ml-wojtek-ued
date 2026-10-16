"""Final-eval tables for report.md from the wandb eval runs of deploy-warmstart.

uv run python experiments/deploy-warmstart/tables.py > tables.md
"""

import wandb

PROJECT = "dominikgalus9-wroc-aw-university-of-science-and-technology/wojtyla"
NATIVE = {
    "upstream 305M, imported (no fine-tune)": "deploy_ws_baseline_a1long_final_eval",
    "upstream 1.5B, imported (no fine-tune)": (
        "deploy_ws_baseline_upstream_1p5b_final_eval"
    ),
    "1.5B + ACCEL 150M, seed 0": "deploy_ws_accel_seed0_final_eval",
    "1.5B + ACCEL 150M, seed 1": "deploy_ws_accel_seed1_final_eval",
    "1.5B + ACCEL 150M, seed 2": "deploy_ws_accel_seed2_final_eval",
}
V10BOX = {
    "v10 seed4 (paper eval: v10 commands, 2048 levels)": (
        "v10_wojtek_accel_seed4_final_eval"
    ),
    "v10 seed4": "v10_wojtek_accel_seed4_v10box_eval",
    "upstream 1.5B, imported": "deploy_ws_baseline_upstream_1p5b_v10box_eval",
    "1.5B + ACCEL, seed 0": "deploy_ws_accel_seed0_v10box_eval",
    "1.5B + ACCEL, seed 1": "deploy_ws_accel_seed1_v10box_eval",
    "1.5B + ACCEL, seed 2": "deploy_ws_accel_seed2_v10box_eval",
}
COLUMNS = [
    ("d0.25 success", "random/d0.25/success_rate"),
    ("d0.25 cmd", "random/d0.25/cmd_success_rate"),
    ("d0.25 fall", "random/d0.25/fall_rate"),
    ("d1.0 success", "random/d1.0/success_rate"),
    ("d1.0 cmd", "random/d1.0/cmd_success_rate"),
    ("d1.0 fall", "random/d1.0/fall_rate"),
    ("d1.0 travel m", "random/d1.0/mean_travel"),
    ("uniform success", "param/uniform/success_rate"),
    ("suite success", "suite/success_rate"),
    ("flat success", "flat/success_rate"),
]


def summary(api: wandb.Api, name: str) -> dict:
    runs = [
        r
        for r in api.runs(PROJECT, filters={"display_name": name})
        if r.state == "finished"
    ]
    return dict(max(runs, key=lambda r: r.created_at).summary) if runs else {}


def table(api: wandb.Api, title: str, rows: dict[str, str]) -> None:
    print(f"\n### {title}\n")
    print("| policy | " + " | ".join(c for c, _ in COLUMNS) + " |")
    print("|---" * (len(COLUMNS) + 1) + "|")
    for label, name in rows.items():
        s = summary(api, name)
        cells = ["-" if s.get(key) is None else f"{s[key]:.3f}" for _, key in COLUMNS]
        print(f"| {label} | " + " | ".join(cells) + " |")


def main() -> None:
    api = wandb.Api(timeout=120)
    table(api, "Native deploy eval (deploy arena, deploy command mix)", NATIVE)
    table(api, "v10box: v10 6x6 m arena, forward command in both boxes", V10BOX)


if __name__ == "__main__":
    main()
