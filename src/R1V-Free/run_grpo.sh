cd src/R1V-Free

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export REWARD_GPUS=<GPU_NUMS of REWARD_MODEL>

torchrun --nproc_per_node="6" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/grpo.py \
    --output_dir <OUTPUT_DIR> \
    --model_name_or_path <PATH-TO-Qwen2-VL-2B-Instruct> \
    --dataset_name <PATH-TO-DATASET> \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 10 \
    --run_name Qwen2-VL-2B-GRPO-RLHF-V \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8
