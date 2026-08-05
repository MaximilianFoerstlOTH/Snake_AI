import sys
from pathlib import Path

from rl_zoo3.train import train

DEFAULT_ALGO = "ppo"
HYPERPARAMS_DIR = Path(__file__).parent / "hyperparams"


def resolve_algo(argv: list[str]) -> str:
    algo = DEFAULT_ALGO
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--algo" and i + 1 < len(argv):
            algo = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--algo="):
            algo = arg.split("=", 1)[1]
        i += 1
    return algo


if __name__ == "__main__":
    algo = resolve_algo(sys.argv)
    conf_file = HYPERPARAMS_DIR / f"{algo}.yml"
    sys.argv[1:1] = [
        "--algo",
        DEFAULT_ALGO,
        "--env",
        "snake-v0",
        "--gym-packages",
        "game",
        "--conf-file",
        str(conf_file),
        "-tb",
        "logs/tb",
    ]
    train()
