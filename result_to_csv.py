import json
from typing import Annotated
import tyro
from dataclasses import dataclass
import csv
from tyro.conf import arg


@dataclass
class CLI:
    json_path: Annotated[str, arg(aliases=["-t"])]


if __name__ == "__main__":
    cli = tyro.cli(CLI)

    with open(cli.json_path, "r") as f:
        data = json.load(f)

    possible_train_models = set(["thin_prism_fisheye", "opencv_fisheye", "rad_tan_thin_prism_fisheye"])

    inner_dict = data["15000"]
    eval_models = set(inner_dict.keys())
    train_model = possible_train_models.intersection(eval_models).pop()
    metric = "PSNR"

    rows = []
    rows.append(["train \ eval"] + list(eval_models))
    rows.append([train_model] + [f"{inner_dict[eval_model][metric]:.2f}" for eval_model in eval_models]) 

    # save csv
    # find json directory and save there
    csv_path = cli.json_path.replace(".json", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
