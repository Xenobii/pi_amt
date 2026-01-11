import os
import hydra
import mir_eval
from hydra.utils import instantiate
from omegaconf import DictConfig



def eval_model(
    ref_intervals,
    ref_piches, 
    est_intervals,
    est_pitches,
    onset_tolerance=0.05,
    pitch_tolerance=50.0,
    offset_ratio=None
):
    scores = mir_eval.transcription.evaluate(
        ref_intervals,
        ref_piches, 
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

    # Load model
    model_cfg = cfg.models[0]
    model = instantiate(model_cfg)
    print(f"Loaded model: {model.name}")

    # Load dataset
    dataset_cfg = cfg.datasets[0]
    dataset = instantiate(dataset_cfg)
    print(f"Loaded dataset: {dataset.name}")
    
    item = dataset[0]

    # Load permuter
    permutation_cfg = cfg.permutations[1]
    permutation = instantiate(permutation_cfg)
    print(f"Loaded permuter: {permutation.name}")

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

    precision    = scores.get("Precision")
    recall       = scores.get("Recall")
    f_measure    = scores.get("F-measure")

    precision_no = scores.get("Precision_no_offset")
    recall_no    = scores.get("Recall_no_offset")
    f_measure_no = scores.get("F-measure_no_offset")

    onset_precision = scores.get("Onset_Precision")
    onset_recall    = scores.get("Onset_Recal")
    onset_f_measure = scores.get("Onset_F-measure")

    # Write MIDI (optional)
    file_path = os.path.join("./test_data", "test_mid.mid")
    model.write_midi(output, file_path)

    print(f"\nPrediction successful!")



if __name__ == "__main__":
    prediction_demo()