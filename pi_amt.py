# Permutation importance in Automatic Music Transcription
# Alvanos Stelios
# steliosalvanos@gmail.com

import numpy as np
import json
import hydra
from pathlib import Path
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig
from collections import defaultdict
import logging

from evaluation import eval_model


log = logging.getLogger(__name__)


def save_metrics(scores: dict, model_name: str, permutation_name: str, dataset_name: str):
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_file = run_dir / "metrics.json"

    out_data = {
        "model": model_name,
        "dataset": dataset_name,
        "permutation": permutation_name,
        "scores": scores
    }

    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)



@hydra.main(version_base="1.3", config_path="config", config_name="config")
def evaluate_all(cfg: DictConfig):
    log.info("-- Permutation Importance in Multi-Pitch Estimation --")
    
    # Load modules
    model       = instantiate(cfg.model)
    dataset     = instantiate(cfg.dataset)
    permutation = instantiate(cfg.permutation)

    log.info(f"Model    : {model.name}")
    log.info(f"Dataset  : {dataset.name}")
    log.info(f"Permuter : {permutation.name}")

    # Load model
    model.load()
    model.load_hook(permutation)
    metrics = defaultdict(list)
    
    # Evalutation pipeline
    subsample_every = cfg.evalutation.subsample_every
    for idx, item in enumerate(tqdm(
        dataset,
        desc=f"Evalutating for {model.name}, {dataset.name}, {permutation.name}"
    )):
        # Sample one in 4 (validation size = 25%)
        if idx % subsample_every != 0:
            continue

        # Handle masks (if necessary)
        if hasattr(permutation, "target"):
            target = model.create_midi_target(item["mid_file"])
            permutation.load_target(target)

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

    log.info("\nEvaluation results:")
    for k, v in avg_scores.items():
        log.info(f"    {k}: {round(v, 4)}")

    save_metrics(avg_scores, model.name, permutation.name, dataset.name)

    log.info("-- Evaluation Successful! --\n")


if __name__=="__main__":
    evaluate_all()
