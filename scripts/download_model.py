#!/usr/bin/env python3
"""
Download the DeepSeek-R1-Distill-Qwen-1.5B model from Hugging Face
to the path specified by the environment variable MODEL_DIR.
"""

import argparse
import os
import sys

from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--local_dir", default="/models/ds-r1-distill-qwen-1.5b")

    args = parser.parse_args()

    hf_repo = args.hf_repo
    local_dir = Path(os.path.expanduser(args.local_dir))

    if local_dir.exists():
        print(f"[INFO] Model {hf_repo} already exists at {local_dir}, skipping download.")

        return 0

    print(f"[INFO] Downloading model {hf_repo} to {local_dir} ...")

    local_dir.mkdir(parents=True)

    try:
        snapshot_download(
            repo_id=hf_repo,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # safer when mounting into Docker
        )
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Model {hf_repo} available at {local_dir}")


if __name__ == "__main__":
    main()
