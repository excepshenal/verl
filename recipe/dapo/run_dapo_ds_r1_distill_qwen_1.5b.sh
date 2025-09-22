#!/usr/bin/env bash
set -xeuo pipefail

# Params from "Scaling Up RL" paper unless otherwise specified

# Data

# Originally set to 1024, but I hit a prompt with length 3317
max_prompt_length=8096
# "Scaling Up RL" sets context window limit at 8096 (not a power of 2?),
# not sure if they meant max_response_length or max_prompt_length + max_response_length
max_response_length=8096

# Actor

train_prompt_bsz=256
train_prompt_mini_bsz=64
# use dynamic micro batch size
ppo_max_token_len_per_gpu=16384 # from verl example dapo scripts

clip_ratio_low=0.2
clip_ratio_high=0.4

use_kl_loss=True
kl_loss_coef=0.0001

actor_optim_lr=2e-6

# Rollout

n_resp_per_prompt=16
temperature=1.2

# actor params should hopefully fit in remaining 35% of H100 given it's 1.5b model.
# this is within range of verl/docs/perf/perf_tuning.rst, but may be risky
gpu_memory_utilization=0.65
rollout_disable_log_stats=False # per verl/docs/perf/perf_tuning.rst
enable_chunked_prefill=False # prompts are short
tensor_model_parallel_size=1 # 1.5b model doesn't need TP

# Training

# initial attempt
trainer_nnodes=1
trainer_n_gpus_per_node=8
trainer_epochs=1 # DeepScaleR contains ~40k rows; with batches of 256, each epoch is around 150 training steps
trainer_log_val_generations=5
trainer_save_freq=100 # save model every 100 steps
trainer_test_freq=5 # eval model every 5 steps
trainer_logger='["console","wandb"]'

# Filepaths

project_name='dapo'
exp_name='ds-r1-distill-qwen-1.5b-exp-0'

PROJECT_DIR=${PROJECT_DIR:-"/workspace/verl"}
MODEL_PATH=${MODEL_PATH:-"/models/ds-r1-distill-qwen-1.5b"}
TRAIN_FILE=${TRAIN_FILE:-"/data/deepscaler/train.parquet"}
TEST_FILE=${TEST_FILE:-"/data/aime24/test.parquet"}

RUNTIME_ENV=${RUNTIME_ENV:-"${PROJECT_DIR}/recipe/dapo/runtime_env.yaml"}

# Defaults are set in verl/recipe/dapo/config/dapo_trainer.yaml,
# which itself references verl/trainer/config/ppo_trainer.yaml.
# Some defaults include:
# - actor_rollout_ref.actor.strategy: fsdp
# - fsdp offload params: False (unnecessary for 1.5b model)
# - actor_rollout_ref.rollout.val_kwargs.do_sample: False
#   ("Scaling Up RL" samples for final eval, but doesn't clarify how they eval during training)
ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.optim.lr=${actor_optim_lr} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.disable_log_stats=${rollout_disable_log_stats} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    reward_model.reward_manager=dapo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=${PROJECT_DIR}/recipe/dapo/rllm_reward.py \
    custom_reward_function.name=rllm_reward_fn_math_transformed \
    trainer.nnodes=${trainer_nnodes} \
    trainer.n_gpus_per_node=${trainer_n_gpus_per_node} \
    trainer.total_epochs=${trainer_epochs} \
    trainer.log_val_generations=${trainer_log_val_generations} \
    trainer.save_freq=${trainer_save_freq} \
    trainer.test_freq=${trainer_test_freq} \
    trainer.logger=${trainer_logger} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name}
