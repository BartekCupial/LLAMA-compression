# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import os
import fire
import time
import psutil
import contextlib

from pathlib import Path

import arithmeticcoding

from tqdm.auto import tqdm
from datasets import load_dataset

from llama import Tokenizer


def compress(tokens, freqs, bitout):
    enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
    for i in tqdm(range(len(tokens))):
        symbol = tokens[i]
        enc.write(freqs, symbol)
        freqs.increment(symbol)
    enc.finish()


def decompress(freqs, bitin, tokenizer, out):
    dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
    decoded_tokens = [] 
    while True:
        symbol = dec.read(freqs)
        if symbol == 2: 
            break
        decoded_tokens.append(symbol)
        freqs.increment(symbol)

    decoded_data = tokenizer.decode(decoded_tokens)
    out.write(decoded_data)
    return decoded_data


def main(
    tokenizer_path: str, 
    enc_dir: str = "arithmetic_enc", 
    dec_dir: str = "arithmetic_dec",
    n_files: int = 200,
    compress_only: bool = True,
):
    mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    wiki = load_dataset("wikipedia", "20220301.en", split="train")
    mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"RAM memory used: {(mem_after - mem_before)} MB")

    enc_dir = Path(enc_dir)
    dec_dir = Path(dec_dir)
    enc_dir.mkdir(parents=True, exist_ok=True)
    dec_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(model_path=tokenizer_path)
    initfreqs = arithmeticcoding.FlatFrequencyTable(tokenizer.sp_model.vocab_size())
    freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
    dec_freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)

    for i in range(n_files):
        text = wiki[i]["text"]

        compressed_name = f"compressed_{i}.bin"
        decompressed_name = f"decompressed_{i}.txt"

        tokens = tokenizer.encode(text, bos=True, eos=True)

        start_time = time.time()
        with contextlib.closing(arithmeticcoding.BitOutputStream(open(enc_dir / compressed_name, "wb"))) as bitout:
            compress(tokens, freqs, bitout)
        
        if not compress_only:
            with open(enc_dir / compressed_name, "rb") as inp, open(dec_dir / decompressed_name, "w") as out:
                bitin = arithmeticcoding.BitInputStream(inp)
                decompress(dec_freqs, bitin, tokenizer, out)

        print(f"Processed text {i} in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    fire.Fire(main)
