# Copyright (c) 2023, Tri Dao.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/forward_step.py#L31
import gc
import time
from dataclasses import dataclass, field
from typing import Optional, Union, Sequence, Callable

import torch
from torch import Tensor, nn
from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
from transformers import BeamSearchScorer
from transformers import GPT2Tokenizer
from transformers.generation.logits_process import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""
    max_sequence_len: int
    max_batch_size: int
    sequence_len_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    fused_ft_kernel: bool = False
    lengths_per_sample: Optional[Tensor] = None


def torch_int_div(tensor1, tensor2):
    """
    A function that performs integer division across different versions of PyTorch.
    """
    return torch.div(tensor1, tensor2, rounding_mode="floor")


def is_ngram_blocked(input_tokens, next_token, n=4):
    for i in range(len(input_tokens) - n + 1):
        # if the ngram is repeated, then block
        if all(input_tokens[i:i + n] == torch.cat([input_tokens[-n + 1:], next_token])):
            return True
    return False


def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf."""
    if top_p <= 0.0:
        return
    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float('-inf'))


def sample(logits, top_k=1, top_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, 'top-p should be in (0, 1].'
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            logits_top /= temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1)
            ]
        else:
            logits_top = logits / temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1)


def run_model(model, input_ids, inference_params, batch_size, beam_width, vocab_size, beam_scores):
    logits = model(input_ids.view(batch_size * beam_width, -1),
                   inference_params=inference_params)[0].logits[:, -1]

    if vocab_size is not None:
        logits = logits[..., :vocab_size]
    else:
        vocab_size = logits.shape[-1]

    # next_tokens = []
    # for i in range(2 * beam_width):
    #     next_tokens.append(sample(logits.reshape(batch_size, -1), top_k=50, top_p=0.9, temperature=1.0))
    # next_tokens = torch.stack(next_tokens, dim=1)  # [batch_size, num_beams]
    # next_token_scores = torch.zeros_like(next_tokens)

    next_token_scores = torch.nn.functional.log_softmax(logits, dim=-1)  # [B * beam_width, vocab_size]
    next_token_scores = next_token_scores + beam_scores.view(-1, 1).expand_as(next_token_scores)

    # reshape
    next_token_scores = next_token_scores.view(batch_size, beam_width * vocab_size)
    next_token_scores, next_tokens = torch.topk(next_token_scores,
                                                4 * beam_width,
                                                dim=1,
                                                largest=True,
                                                sorted=True)

    next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
    next_tokens = next_tokens % vocab_size

    return next_indices, next_tokens, next_token_scores


def decode(input_ids, model, max_length, top_k=1, top_p=0.0, temperature=1.0,
           eos_token_id=None, vocab_size=None, tensor_parallel=1, fused_ft_kernel=False,
           cg=False, timing=False):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    # beam search parameters
    num_beams = 5
    num_beam_groups = 1
    batch_size, seqlen_og = input_ids.shape
    device = input_ids.device

    logits_warper = LogitsProcessorList()
    if temperature != 1.0:
        logits_warper.append(TemperatureLogitsWarper(temperature=temperature))
    if top_k > 0:
        logits_warper.append(TopKLogitsWarper(k=top_k))

    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        device=device,
        num_beam_groups=num_beam_groups,
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams
    num_beam_groups = beam_scorer.num_beam_groups
    num_sub_beams = num_beams // num_beam_groups
    beam_indices = None

    beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
    beam_scores[:, ::num_sub_beams] = 0
    beam_scores = beam_scores.view((batch_size * num_beams,))

    # extend input_ids to beam_width
    input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1).reshape(batch_size * num_beams, -1)
    if cg:
        assert fused_ft_kernel
        if not hasattr(model, '_decoding_cache'):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(
            model, model._decoding_cache, batch_size * num_beams, seqlen_og, max_length,
            tensor_parallel=tensor_parallel
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.max_sequence_len = max_length
        inference_params.max_batch_size = batch_size * num_beams
        inference_params.sequence_len_offset = 0
    else:
        inference_params = InferenceParams(max_sequence_len=max_length, max_batch_size=batch_size * num_beams,
                                           fused_ft_kernel=fused_ft_kernel)

    with torch.inference_mode():
        if timing:
            torch.cuda.synchronize()
            start = time.time()
        cur_len = seqlen_og

        while True:
            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            model_inputs = input_ids
            outputs = model(model_inputs, inference_params=inference_params)
            logits = outputs[0].logits[:, -1]

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]
                next_token_logits = logits[batch_group_indices, :]
                vocab_size = next_token_logits.shape[-1]

                # sample
                next_token_scores = nn.functional.log_softmax(
                    next_token_logits, dim=-1
                )  # (batch_size * num_beams, vocab_size)
                next_token_scores = next_token_scores + beam_scores[batch_group_indices].unsqueeze(-1)
                next_token_scores = logits_warper(input_ids, next_token_scores)
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                probs = nn.functional.softmax(next_token_scores, dim=-1)

                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)

                # # beam
                # next_token_scores_processed = next_token_logits   # FIXME (@seungjuh)
                # next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                # next_token_scores = next_token_scores.expand_as(next_token_scores_processed)
                #
                # # reshape for beam search
                # next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)
                # next_token_scores, next_tokens = torch.topk(
                #     next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True)

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
                next_tokens = next_tokens % vocab_size

                # ngram block
                for batch_idx in range(batch_size):
                    for beam_token_rank, (next_token, next_index) in enumerate(
                            zip(next_tokens[batch_idx], next_indices[batch_idx])
                    ):
                        batch_beam_idx = batch_idx * group_size + next_index
                        context_tokens = group_input_ids[batch_beam_idx, :]
                        if is_ngram_blocked(context_tokens, next_token.unsqueeze(0)):
                            next_token_scores[batch_idx, beam_token_rank] = -1e9

                # stateless
                process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    beam_indices=process_beam_indices,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                        num_beams * torch.div(beam_idx, group_size, rounding_mode="floor") + group_start_idx + (
                        beam_idx % group_size)
                )

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)
            print(tokenizer.decode(input_ids[0][seqlen_og:]))

            cur_len += 1
            # if beam_scorer.is_done or stopping_criteria(input_ids, None):
            if beam_scorer.is_done or cur_len >= max_length:
                break

        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_length,
            beam_indices=final_beam_indices,
        )
        if timing:
            torch.cuda.synchronize()
            print(f'Decoding time: {(time.time() - start) * 1000:.0f}ms')
    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    return output_cls(
        sequences=sequence_outputs['sequences'],
        scores=sequence_outputs['sequence_scores'],
    )


class GenerationMixin:

    def generate(self, input_ids, max_length, top_k=1, top_p=0.0, temperature=1.0,
                 return_dict_in_generate=False, output_scores=False, **kwargs):
        output = decode(input_ids, self, max_length, top_k=top_k, top_p=top_p,
                        temperature=temperature, **kwargs)
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences


def allocate_kv_cache(max_batch_size, max_seqlen, nheads, headdim, layers: Union[int, Sequence],
                      device, dtype=torch.float16):
    assert dtype in [torch.float16, torch.bfloat16, torch.float32]
    packsize = 4 if dtype == torch.float32 else 8
    assert headdim % packsize == 0
    k_cache_shape = (max_batch_size, nheads, headdim // packsize, max_seqlen, packsize)
    v_cache_shape = (max_batch_size, nheads, max_seqlen, headdim)
    if isinstance(layers, int):
        layers = range(layers)
    return {i: (torch.empty(k_cache_shape, device=device, dtype=dtype),
                torch.empty(v_cache_shape, device=device, dtype=dtype))
            for i in layers}


def seqlen_to_seqlen_type(seqlen: int) -> int:
    """Convert sequence length to a seqlen_type.
    This is used to determine which cuda graph to use.
    Arguments:
        seqlen: int
    """
    return 0 if seqlen < 32 else (1 if seqlen < 2048 else 2)


def seqlen_type_to_seqlen(seqlen_type: int) -> int:
    assert seqlen_type in [0, 1, 2]
    return 1 if seqlen_type == 0 else (32 if seqlen_type == 1 else 2048)


@dataclass
class DecodingCGCache:
    max_batch_size: int = 0
    max_seqlen: int = 0
    device = None
    dtype = None
    callables: dict = field(default_factory=dict)
    mempool = None
    inference_params: Optional[InferenceParams] = None
    run: Optional[Callable] = None


@torch.inference_mode()
def update_graph_cache(model, cache, batch_size, seqlen_og, max_seqlen, tensor_parallel=1,
                       dtype=None):
    if cache is None:
        cache = DecodingCGCache()
    param_example = next(iter(model.parameters()))
    device = param_example.device
    if dtype is None:
        dtype = param_example.dtype
    if ((device, dtype) != (cache.device, cache.dtype) or batch_size > cache.max_batch_size
            or max_seqlen > cache.max_seqlen):  # Invalidate the cache
        cache.callables = {}
        cache.mempool = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = batch_size, max_seqlen
        headdim = getattr(model.config, 'head_dim',
                          model.config.hidden_size // model.config.num_attention_heads)
        kv_cache = allocate_kv_cache(
            batch_size, max_seqlen, model.config.num_attention_heads // tensor_parallel, headdim,
            model.config.num_hidden_layers, device, dtype
        )
        lengths_per_sample = torch.full((batch_size,), seqlen_og, dtype=torch.int32, device=device)
        cache.inference_params = InferenceParams(
            max_sequence_len=max_seqlen, max_batch_size=batch_size,
            sequence_len_offset=seqlen_og, key_value_memory_dict=kv_cache, fused_ft_kernel=True,
            lengths_per_sample=lengths_per_sample
        )
        cache.mempool = torch.cuda.graphs.graph_pool_handle()
    for s_type in range(seqlen_to_seqlen_type(seqlen_og), seqlen_to_seqlen_type(max_seqlen) + 1):
        if s_type not in cache.callables:
            seqlen = min(max(seqlen_og, seqlen_type_to_seqlen(s_type)), max_seqlen)
            cache.callables[s_type] = capture_graph(
                model, cache.inference_params, batch_size, seqlen_og, seqlen, mempool=cache.mempool
            )

    def dispatch(input_ids, position_ids, seqlen):
        return cache.callables[seqlen_to_seqlen_type(seqlen)](input_ids, position_ids, seqlen)

    cache.run = dispatch
    cache.inference_params.sequence_length_offset = 0  # Reset so it's not confusing
    return cache


def capture_graph(model, inference_params, batch_size, seqlen_og, max_seqlen, mempool=None):
    assert max_seqlen >= seqlen_og
    device = next(iter(model.parameters())).device
    input_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)
    position_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)
    inference_params.lengths_per_sample[:] = seqlen_og

    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            logits = model(input_ids, position_ids=position_ids,
                           inference_params=inference_params).logits[:, -1]
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)
    # Captures the graph
    # To allow capture, automatically sets a side stream as the current stream in the context
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        logits = model(input_ids, position_ids=position_ids,
                       inference_params=inference_params).logits[:, -1]

    def run(new_input_ids, new_position_ids, seqlen):
        inference_params.lengths_per_sample[:] = seqlen
        input_ids.copy_(new_input_ids)
        position_ids.copy_(new_position_ids)
        graph.replay()
        return logits

    return run
