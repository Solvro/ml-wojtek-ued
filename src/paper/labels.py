ROBOT_LABEL = {
    "wojtek": "W01-TEK",
    "go1": "Go1",
    "barkour": "Barkour vB",
    "spot": "Spot",
    "go2": "Go2",
}
METHODS = {
    "dr": "DR",
    "dr_easy": "DR-easy",
    "dr_mixed": "DR-mixed",
    "plr": r"PLR$^{\perp}$",
    "sfl": "SFL",
    "accel": "ACCEL",
    "accel_grow": "ACCEL-grow",
    "accel_sfl": "ACCEL-SFL",
    "paired": "PAIRED",
    "paired_uni": "PAIRED+UniBC",
    "paired_bi": "PAIRED+BiBC",
    "paired_evo": "PAIRED+Evo",
}


def method_label(method: str) -> str:
    return METHODS.get(method, method.replace("_", r"\_"))


def method_order(methods: set[str]) -> list[str]:
    known = [m for m in METHODS if m in methods]
    return known + sorted(methods - set(METHODS))
