# Permutation importance in Automatic Music Transcription
# Alvanos Stelios
# steliosalvanos@gmail.com

import numpy as np
import pandas as pd
import json
import hydra
from pathlib import Path
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig
from collections import defaultdict

from evaluation import eval_model


def save_metrics(scores: dict):
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_file = run_dir / "metrics.json"

    with open(out_file, "w") as f:
        json.dump(scores, f, indent=2)



@hydra.main(version_base="1.3", config_path="config", config_name="config")
def evaluate_all(cfg: DictConfig):
    
    # Load modules
    model       = instantiate(cfg.model)
    dataset     = instantiate(cfg.dataset)
    permutation = instantiate(cfg.permutation)

    print(f"Model    : {model.name}")
    print(f"Dataset  : {dataset.name}")
    print(f"Permuter : {permutation.name}")

    # Load model
    model.load()
    model.load_hook(permutation)

    metrics = defaultdict(list)
    
    # Evalutation pipeline
    for item in tqdm(dataset, desc=f"Evalutating for {model.name}, {dataset.name}, {permutation.name}"):
        output = model.predict(item["wav_file"])

        # Evaluate
        ref_intervals, ref_pitches = dataset.create_eval_data(item["mid_file"])
        est_intervals, est_pitches = model.prepare_for_eval(output)

        scores = eval_model(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches
        )

        for k, v in scores.items():
            metrics[k].append(v)

    # Clear hooks
    model.clear_hooks()

    avg_scores = {k: np.mean(vs) for k, vs in metrics.items()}

    print("\nEvaluation results:")
    for k, v in avg_scores.items():
        print(f"    {k}: {round(v, 4)}")

    save_metrics(avg_scores)



if __name__=="__main__":
    print("\n-- Permutation Importance in Multi-Pitch Estimation --\n")
    evaluate_all()
    print("\n-- Evaluation Successful! --\n")
