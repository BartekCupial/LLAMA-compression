# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch
import numpy as np

import arithmeticcoding

from tqdm.auto import tqdm
from llama.tokenizer import Tokenizer
from llama.model import Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer, freq_mult: float = 20000):
        self.model = model
        self.tokenizer = tokenizer
        self.freq_mult = freq_mult

    def encode(
        self,
        prompts: List[str],
        bitout,
        temperature: float = 0.8,
    ) -> List[str]:
        enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

        bsz = len(prompts)
        assert bsz == 1
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=True) for x in prompts]

        max_prompt_size = max([len(t) for t in prompt_tokens])

        tokens = torch.full((bsz, max_prompt_size), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        start_pos = 0 # we want to start from the beginning
        prev_pos = 0
        # saved_token_probs = []
        for cur_pos in tqdm(range(start_pos + 1, max_prompt_size)):
            logits = self.model.forward(tokens[:, max(cur_pos - params.max_seq_len, 0):cur_pos], 0)
            probs = torch.softmax(logits / temperature, dim=-1)

            raw_frequencies = (probs[0].detach().cpu().numpy() * self.freq_mult).astype(int)
            raw_frequencies[raw_frequencies == 0] = 1
            freqs = arithmeticcoding.SimpleFrequencyTable(raw_frequencies)
            enc.write(freqs, tokens[0][cur_pos].item())

            # saved_token_probs.append(probs)
            next_token = tokens[:, cur_pos]
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
        enc.finish()

    def decode(self, bitin, out, temperature: float = 0.8):
        dec = arithmeticcoding.ArithmeticDecoder(32, bitin)

        decoded_tokens = [] 
        bsz = 1
        params = self.model.params
        total_len = params.max_seq_len
        prompt_tokens = [self.tokenizer.encode("", bos=True, eos=False)]

        tokens = torch.full((bsz, 1), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()

        prev_pos = 0
        cur_pos = 1
        while True:
            logits = self.model.forward(tokens[:, max(cur_pos - total_len, 0):cur_pos], 0)
            probs = torch.softmax(logits / temperature, dim=-1)

            raw_frequencies = (probs[0].detach().cpu().numpy() * self.freq_mult).astype(int)
            raw_frequencies[raw_frequencies == 0] = 1
            freqs = arithmeticcoding.SimpleFrequencyTable(raw_frequencies)

            symbol = dec.read(freqs)

            if symbol == 2: 
                break
            
            decoded_tokens.append(symbol)
            next_token = torch.tensor([[symbol]]).to(tokens.device)
            tokens = torch.cat([tokens, next_token], dim=1)

            cur_pos += 1

        decoded_data = self.tokenizer.decode(tokens.tolist()[0])
        out.write(decoded_data)
        print(decoded_data)
        return decoded_data

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
