## Quickstart

**Steps 2-6 and 8 apply only to the Ray head node.**

1. On each of your training nodes, pull and run the `veRL` Docker image, mounting dirs:

```bash
docker pull verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2
docker run -it \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /data:/data -v /models:/models \
  verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2 \
  bash
```

2. Install the `rllm` dependency:

```bash
git clone https://github.com/excepshenal/rllm.git
cd rllm
git switch math-reward-no-llm
pip install -e . --no-deps
cd ..
```

3. Clone this fork of the `veRL` repo:

```bash
git clone https://github.com/excepshenal/verl.git
cd verl
```

4. Download the model:

```bash
python3 scripts/download_model.py
```

5. Prepare the data:

```bash
python3 examples/data_preprocess/deepscaler.py
python3 examples/data_preprocess/aime24.py
```

6. Get `envsubst`:

```bash
sed -i -E 's|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|http://archive.ubuntu.com/ubuntu/|g' /etc/apt/sources.list
apt-get update && apt-get install -y gettext-base
```

7. Start or join the Ray cluster:

```bash
ray start --head
```

or

```bash
ray start --address='<head_node_ip>:6379'
```

8. Submit the job to the Ray cluster:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
bash recipe/dapo/run_dapo_ds_r1_distill_qwen_1.5b.sh
```
