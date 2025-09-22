"""
Preprocess the AIME24 dataset to parquet format
"""

import argparse
import json
import os

from pathlib import Path
from datasets import load_dataset


def main():
    data_source = "math-ai/aime24"

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/data/aime24")

    args = parser.parse_args()

    local_dir = Path(os.path.expanduser(args.local_dir))
    if local_dir.exists():
        print(f"[INFO] Dataset {data_source} already exists at {local_dir}, skipping download.")

        return 0

    local_dir.mkdir(parents=True)

    print(f"[INFO] Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, trust_remote_code=True)

    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")
            question = question + " " + instruction_following
            solution = example.pop("solution")

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }

            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    # Save one example as JSON for reference
    example = test_dataset[0]
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=2)

    print(f"[INFO] Dataset is available at {local_dir}")


if __name__ == "__main__":
    main()
