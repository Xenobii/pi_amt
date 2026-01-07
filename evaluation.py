import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig



def eval_model(
    model,
    dataset,
    permutation = None,
):
    return NotImplementedError


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
    model.predict(item["wav_file"])

    print(f"\nPrediction successful!")



if __name__ == "__main__":
    prediction_demo()