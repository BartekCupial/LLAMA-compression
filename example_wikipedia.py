# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import psutil
import contextlib

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
import debugpy

from datasets import load_dataset
import arithmeticcoding

# debugpy.listen(('localhost', 5678))
# debugpy.wait_for_client()



def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    freq_mult: float = 20000,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer, freq_mult=freq_mult)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    freq_mult: float = 20000,
    enc_dir: str = "comp", 
    dec_dir: str = "decomp",
    n_files: int = 200,
    compress_only: bool = True,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    wiki = load_dataset("wikipedia", "20220301.en", split="train")
    mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"RAM memory used: {(mem_after - mem_before)} MB")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size, freq_mult=freq_mult
    )

    enc_dir = Path(enc_dir)
    dec_dir = Path(dec_dir)
    enc_dir.mkdir(parents=True, exist_ok=True)
    dec_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_files):
        text = wiki[i]["text"]

        compressed_name = f"compressed_{i}.bin"
        decompressed_name = f"decompressed_{i}.txt"

        start_time = time.time()
        with contextlib.closing(arithmeticcoding.BitOutputStream(open(enc_dir / compressed_name, "wb"))) as bitout:
            generator.encode([text], bitout, temperature=temperature)
        
        if not compress_only:
            with open(enc_dir / compressed_name, "rb") as inp, open(dec_dir / decompressed_name, "w") as out:
                bitin = arithmeticcoding.BitInputStream(inp)
                generator.decode(bitin, out, temperature=temperature)

        print(f"Processed text {i} in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    fire.Fire(main)
