import os
import yaml
from pathlib import Path

from src.loader import DataLoader
from src.solver import Solver


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_no"])

    mode = config["mode"]

    save_path = Path(config["training"]["save_path"]) / config["version"]
    save_path.mkdir(parents=True, exist_ok=True)
    config["save_path"] = save_path
    
    datasets = DataLoader(
        mode,
        **config["dataset"]
    )
    solver = Solver(config, datasets)

    if mode == "train":
        solver.train()
    if mode == "test":
        solver.test()

if __name__ == "__main__":
    with open("./config.yaml") as f:
        config = yaml.load(f)

    main(config)
