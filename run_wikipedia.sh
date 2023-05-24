#!/bin/bash

#SBATCH --job-name=LLAMA
#SBATCH --output=logger-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --qos=big  # test (1 GPU, 1 hour), quick (1 GPU, 1 day), normal (2 GPU, 2 days), big (4 GPU, 7 days)
#SBATCH --partition=dgxmatinf  # dgxa100 (mini-servers), dgxteam (dgx1 for Team-Net), dgxmatinf (dgx2 for WMiI), dgx (dgx1 and dgx2)

#squeue -u ${USER} --Format "JobID:.6 ,Partition:.4 ,Name:.10 ,StateCompact:.2 ,TimeUsed:.11 ,Qos:.7 ,TimeLeft:.11 ,ReasonList:.16 ,Command:.40"

# linking libraries enable running on arbitrary servers


# torchrun --nproc_per_node 1 example_wikipedia.py \
#     --ckpt_dir /home/z1188643/llama-dl/7B \
#     --tokenizer_path /home/z1188643/llama-dl/tokenizer.model \
#     --freq_mult 2000 \
#     --enc_dir comp2000 \
#     --dec_dir decomp2000 \
#     --n_files 50

# torchrun --nproc_per_node 1 example_wikipedia.py \
#     --ckpt_dir /home/z1188643/llama-dl/7B \
#     --tokenizer_path /home/z1188643/llama-dl/tokenizer.model \
#     --freq_mult 2000 \
#     --enc_dir comp20000 \
#     --dec_dir decomp20000 \
#     --n_files 50

torchrun --nproc_per_node 1 example_wikipedia.py \
    --ckpt_dir /home/z1188643/llama-dl/7B \
    --tokenizer_path /home/z1188643/llama-dl/tokenizer.model \
    --freq_mult 200000 \
    --enc_dir comp200000 \
    --dec_dir decomp200000 \
    --n_files 50
