import sys
import game
from rl_zoo3.enjoy import enjoy

if __name__ == "__main__":
    # Insert defaults at the front so user-supplied args take precedence.
    # Override with e.g. `python enjoy.py --algo dqn`.
    sys.argv[1:1] = [
        "--algo", "ppo",
        "--env", "snake-v0",
        "--folder", "logs",
        #"--no-render"
    ]
    enjoy()
 