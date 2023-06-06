import os
import sys
import fire
import time
import contextlib

import arithmeticcoding

from example_wikipedia import setup_model_parallel, load


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    input_file: str = "enwik6.txt",
    output_file: str = "enwik6.bin",
    temperature: float = 0.8,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    freq_mult: float = 20000,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size, freq_mult=freq_mult
    )

    with open(input_file, "r") as input, contextlib.closing(arithmeticcoding.BitOutputStream(open(output_file, "wb"))) as bitout:
        text = input.read()

        start_time = time.time()
        generator.encode([text], bitout, temperature=temperature)
        print(f"Processed {input_file} in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    fire.Fire(main)
