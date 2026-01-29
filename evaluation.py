import os
import hydra
import mir_eval
from hydra.utils import instantiate
from omegaconf import DictConfig
import logging


log = logging.getLogger(__name__)


def eval_model(
    ref_intervals,
    ref_pitches, 
    est_intervals,
    est_pitches,
    onset_tolerance=0.05,
    pitch_tolerance=50.0,
    offset_ratio=None
):
    # Handle empty prediction
    if est_intervals.shape[0] == 0 or ref_intervals.shape[0] == 0:
        return {
            'Precision_no_offset': 0.0,
            'Recall_no_offset': 0.0,
            'F-measure_no_offset': 0.0,
            'Average_Overlap_Ratio_no_offset': 0.0,
            'Onset_Precision': 0.0,
            'Onset_Recall': 0.0,
            'Onset_F-measure': 0.0
        }

    scores = mir_eval.transcription.evaluate(
        ref_intervals,
        ref_pitches, 
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
        offset_ratio=offset_ratio
    )
    return scores
    

@hydra.main(version_base="1.3", config_path="config", config_name="config")
def permutation_evaluation_demo(cfg: DictConfig):
    """Prediction demo function for bugfixing and testing"""

    log.info("-- Evaluation Demo --")

    # --- model --- 
    model = instantiate(cfg.model)
    log.info(f"Loaded model     : {model.name}")

    # --- adapter ---
    adapter = instantiate(cfg.adapter, model.target_shape)
    log.info(f"Target shape     : {model.target_shape}")

    # --- dataset ---
    dataset = instantiate(cfg.dataset)
    log.info(f"Loaded dataset   : {dataset.name}")

    # --- permutation --- 
    permutation = instantiate(cfg.permutation, adapter=adapter, complex=model.complex)
    log.info(f"Loaded permuter  : {permutation.name}")
    
    item = dataset[1]

    # Pipeline
    model.load()
    model.load_hook(permutation)

    if hasattr(permutation, "target"):
        target = model.create_midi_target(item["mid_file"])
        permutation.load_target(target)

    output = model.predict(item["wav_file"])
    model.clear_hooks()

    # Evaluate
    ref_intervals, ref_pitches = dataset.create_eval_data(item["mid_file"])
    est_intervals, est_pitches = model.prepare_for_eval(output)

    scores = eval_model(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches
    )

    log.info("Evaluation results:")
    for k, v in scores.items():
        log.info(f"    {k}: {round(v, 4)}")

    # Write MIDI (optional)
    file_path = os.path.join("./test_data", "test_mid.mid")
    model.write_midi(output, file_path)

    log.info(f"Prediction successful!")


if __name__ == "__main__":
    permutation_evaluation_demo()