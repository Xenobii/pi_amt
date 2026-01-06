# Permutation importance in Automatic Music Transcription
# Alvanos Stelios
# steliosalvanos@gmail.com

import hydra
from hydra.utils import instantiate
import pandas as pd

from evaluation import eval_model

"""
Pipeline:

for model in models
    for dataset in datasets
        evaluate model
        for permutation in permutations
            evaluate model with permuted data
            end for
        end for
    end for
end for            
"""

@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg):
    print("-- Permutation Importance in Automatic Music Transcription --")
    
    models       = [instantiate(m) for m in cfg.models]
    datasets     = [instantiate(d) for d in cfg.datasets]
    permutations = [instantiate(p) for p in cfg.permutations]
    
    rows = []

    for model in models:
        for dataset in datasets:
            
            metrics = eval_model(model, dataset)
            
            rows.append({
                "model": model.name,
                "dataset": dataset.name,
                "permutation": None,
                **metrics,
            })

            for permutation in permutations:
                metrics = eval_model(model, dataset, permutation)

                rows.append({
                    "model": model.name,
                    "dataset": dataset.name,
                    "permutation": permutation.name,
                    **metrics,
                })
        
    df = pd.DataFrame(rows)
    df.to_parquet("evaluation.parquet")

    print(df.head())



if __name__=="__main__":
    main()
