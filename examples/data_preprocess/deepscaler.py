"""
Preprocess the DeepScaleR-Preview-Dataset to parquet format
"""

import argparse
import json
import os

from pathlib import Path
from datasets import load_dataset


def main():
    data_source = "agentica-org/DeepScaleR-Preview-Dataset"

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/data/deepscaler")

    args = parser.parse_args()

    local_dir = Path(os.path.expanduser(args.local_dir))
    if os.path.exists(local_dir):
        print(f"[INFO] Dataset {data_source} already exists at {local_dir}, skipping download.")

        return 0
    
    local_dir.mkdir(parents=True)

    print(f"[INFO] Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset["train"]
    
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")
            question = question + " " + instruction_following
            answer = example.pop("answer")

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": idx},
            }

            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    # Save one example as JSON for reference
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)

    print(f"[INFO] Dataset is available at {local_dir}")


if __name__ == "__main__":
    main()
