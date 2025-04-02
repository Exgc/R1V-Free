# R1V-Free: Enhancing open-ended understanding of VLLMs with Group Relative Reward

---

### Updates

[//]: # (- 2025-04-02: We achieve sota performance .)
- 2025-04-01: We release the R1V-Free repo.


### For contributors
- Our top development priority is addressing the issues marked with `help wanted` labels, and we welcome ideas/PRs from the community to help solve them.

---


## Setup

```bash
conda create -n r1v-free python=3.11 
conda activate r1v-free


bash setup.sh
```

> [!NOTE] 
> If you meet bug when running the script, first try align your environments with `./src/requirements.txt`


### Supported Models

1. Qwen2-VL
2. Qwen2.5-VL 

### Supported Training Datasets


### Supported Evaluations

#### HallusionBench
We evaluate our models using **HallusionBench**, a diagnostic benchmark designed to measure hallucination and reasoning consistency in visual language models (VLMs). Specifically, we compare the performance of the base model **Qwen2.5-VL-3B-Instruct** and an **RL-enhanced variant (Qwen2.5-3B-RLHF-V)** fine-tuned using **our GRPO pipeline implemented in this project**.

##### GPT Evaluation Results on HallusionBench

| Model                  | Acc per Question (aAcc) | Acc per Question Pair (qAcc) | Acc per Figure (fAcc) | Easy Question Acc | Hard Question Acc |
|------------------------|--------------------------|-------------------------------|------------------------|-------------------|-------------------|
| Qwen2.5-VL-3B-Instruct | 49.42%                   | 17.36%                        | 26.59%                 | 49.23%            | 36.74%            |
| Qwen2.5-3B-RLHF-V | **53.32%**               | **21.32%**                    | **32.95%**             | **49.89%**        | **44.19%**        |

> These results demonstrate that **R1V-Free’s Group Relative Reward (GRPO)** training leads to **significant improvements** in open-ended visual understanding.


## Training

### GRPO

```bash
cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/grpo.py \
    --output_dir ./checkpoint \
    --model_name_or_path /mnt/private_hk/data/Qwen2-VL-2B-Instruct \
    --dataset_name '/mnt/private_hk/data/clevr_cogen_a_train' \
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
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  

```



```bash
cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/grpo.py \
    --output_dir ./checkpoint \
    --model_name_or_path /mnt/private_hk/data/Qwen2-VL-7B-Instruct \
    --dataset_name '/mnt/private_hk/data/clevr_cogen_a_train' \
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
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  

```



```bash
cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node="6" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/grpo.py \
    --output_dir ./checkpoint/RLHF-V \
    --model_name_or_path /mnt/private_hk/data/Qwen2-VL-2B-Instruct \
    --dataset_name '/mnt/private_hk/data/RLHF-V-Dataset-H' \
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
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
```

> [!NOTE] 
> 1. To reproduce the result, keep the per_device_train_batch_size to 1 for now, as there is a revealed bug about batched training. See the [reproduction report](https://github.com/Deep-Agent/R1-V/issues/4#issuecomment-2633348354) here. We realize it is important for effiency and are working on solving it with the community.
> 2. If you meet **OOM Error**, you can try reduce `--num_generations`
> 3. To use vLLM to speed up, please refer to this [script](https://github.com/Deep-Agent/R1-V/blob/main/src/scripts/run_grpo_vllm.sh).


### SFT

We also provide SFT code, please follow the script and edit the config to customize the sft task.

```bash
accelerate launch --config_file src/r1-v/configs/zero2.yaml src/r1-v/src/open_r1/sft.py --config src/r1-v/configs/qwen2vl_sft_config.yaml 
```




## Acknowledgements

We sincerely thank [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) (our initial codebase), [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/), [SuperCLEVR](https://github.com/Lizw14/Super-CLEVR), [G-LLAVA](https://arxiv.org/abs/2312.11370) for providing open source resources and to build the project. Special thanks to [Kimi](https://kimi.moonshot.cn/), [bAInance Labs](https://bainancelabs.com/) for supporting computation resources and [Yuxin Wu](https://scholar.google.com/citations?user=mJQI-gUAAAAJ&hl=en), [Xinyu Zhou](https://scholar.google.com/citations?user=Jv4LCj8AAAAJ&hl=en), [Baobao Chang](https://scholar.google.com.au/citations?user=LaKNyhQAAAAJ&hl=en) for their valuable advice.



[![Star History Chart](https://api.star-history.com/svg?repos=Deep-Agent/R1-V&type=Timeline)](https://star-history.com/#Deep-Agent/R1-V&Timeline)

## Citation

```bib
@misc{chen2025r1v,
  author       = {Cheng Xize, Cai Zhengzhou and Zhao Zhou},
  title        = {R1V-Free: Enhancing open-ended understanding of VLLMs with Group Relative Reward},
  howpublished = {\url{https://github.com/Exgc/R1V-Free}},
  note         = {Accessed: 2025-04-01},
  year         = {2025}
}
```



