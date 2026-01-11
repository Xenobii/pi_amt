import os
import hydra
import mir_eval
from hydra.utils import instantiate
from omegaconf import DictConfig



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
def prediction_demo(cfg: DictConfig):
    """Prediction demo function for bugfixing and testing"""

    print("-- Evaluation Demo --\n")

    # Load with Hydra
    model = instantiate(cfg.model)
    dataset = instantiate(cfg.dataset)
    permutation = instantiate(cfg.permutation)

    print(f"Loaded model: {model.name}")
    print(f"Loaded dataset: {dataset.name}")
    print(f"Loaded permuter: {permutation.name}")
    
    item = dataset[0]

    # Pipeline
    model.load()
    model.load_hook(permutation)
    model.clear_hooks()
    output = model.predict(item["wav_file"])

    # Evaluate
    ref_intervals, ref_pitches = dataset.create_eval_data(item["mid_file"])
    est_intervals, est_pitches = model.prepare_for_eval(output)

    scores = eval_model(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches
    )
    print("\nEvaluation results:")
    for k, v in scores.items():
        print(f"    {k}: {round(v, 4)}")

    # Write MIDI (optional)
    file_path = os.path.join("./test_data", "test_mid.mid")
    model.write_midi(output, file_path)

    print(f"\nPrediction successful!")



if __name__ == "__main__":
    prediction_demo()