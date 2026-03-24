#!/usr/bin/env python3
"""
Full Arditi et al. (2024) pipeline for independent refusal direction extraction.
Adapted for Qwen 2.5/3.5 architecture on FP8 models.

Pipeline:
1. Load model + determine refusal/eoi tokens
2. Filter prompts (keep only harmful that refuse, harmless that don't)
3. Extract mean-diff directions at all layers × eoi positions
4. Evaluate each candidate with 3-point ablation (refusal score, KL div, steering score)
5. Select best direction and save

Reference: "Refusal in Language Models Is Mediated by a Single Direction"
           (Arditi et al., 2024)
"""

import torch
import json
import os
import sys
import math
import contextlib
import functools
import random
import time

from typing import List, Tuple, Callable, Optional
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "Qwen/Qwen3.5-27B"  # BF16 (FP8 not decodable by transformers)
DATA_DIR = "/tmp/arditi"
OUTPUT_DIR = "/tmp/arditi/output"

N_TRAIN = 128        # prompts for direction extraction
N_VAL = 32           # prompts for direction evaluation/selection
BATCH_SIZE = 8       # forward pass batch size
SEED = 42

# Selection thresholds (from paper)
KL_THRESHOLD = 0.1
INDUCE_REFUSAL_THRESHOLD = 0.0
PRUNE_LAYER_PERCENTAGE = 0.20

# Chat template (Qwen 2.5/3.5 — same im_start/im_end format)
CHAT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

# ============================================================
# Hook Utilities (from paper's hook_utils.py)
# ============================================================

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_direction_ablation_input_pre_hook(direction: Tensor):
    """Pre-hook: remove direction from residual stream input to a block."""
    def hook_fn(module, input):
        nonlocal direction
        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input

        direction_normed = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction_normed = direction_normed.to(activation)
        proj = (activation @ direction_normed).unsqueeze(-1) * direction_normed
        activation = activation - proj

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_direction_ablation_output_hook(direction: Tensor):
    """Output hook: remove direction from attn/MLP output."""
    def hook_fn(module, input, output):
        nonlocal direction
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output

        direction_normed = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction_normed = direction_normed.to(activation)
        proj = (activation @ direction_normed).unsqueeze(-1) * direction_normed
        activation = activation - proj

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation
    return hook_fn


def get_activation_addition_input_pre_hook(vector: Tensor, coeff: float):
    """Pre-hook: add scaled vector to residual stream input."""
    def hook_fn(module, input):
        nonlocal vector
        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input

        vector_dev = vector.to(activation)
        activation = activation + coeff * vector_dev

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_mean_activations_pre_hook(layer: int, cache: Tensor, n_samples: int, positions: List[int]):
    """Pre-hook: accumulate mean activations in float64 cache."""
    def hook_fn(module, input):
        activation = input[0].clone().to(cache)
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
    return hook_fn


# ============================================================
# Scoring Functions (from paper's select_direction.py)
# ============================================================

# Token ID for <think> — set at runtime, suppressed from refusal scoring
# because Qwen 3.5 is a reasoning model and <think> dominates both harmful/harmless
THINK_TOKEN_ID = None

def refusal_score(logits: Tensor, refusal_toks: List[int], epsilon: float = 1e-8) -> Tensor:
    """Compute log-odds of refusal tokens at last position.
    Suppresses <think> token to get meaningful refusal/compliance signal."""
    logits = logits.to(torch.float64)
    logits = logits[:, -1, :]  # last position only
    # Suppress <think> token — it dominates both harmful and harmless equally
    if THINK_TOKEN_ID is not None:
        logits[:, THINK_TOKEN_ID] = float('-inf')
    probs = torch.nn.functional.softmax(logits, dim=-1)
    refusal_probs = probs[:, refusal_toks].sum(dim=-1)
    nonrefusal_probs = 1.0 - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)


def kl_div_fn(logits_a: Tensor, logits_b: Tensor, epsilon: float = 1e-6) -> Tensor:
    """KL divergence between two sets of logits. Returns per-sample KL."""
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)
    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)
    kl_divs = torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1)
    return torch.mean(kl_divs, dim=-1)  # mean over sequence positions


# ============================================================
# Helper Functions
# ============================================================

def tokenize_instructions(tokenizer, instructions: List[str]) -> dict:
    """Format and tokenize instructions using Qwen chat template."""
    prompts = [CHAT_TEMPLATE.format(instruction=instr) for instr in instructions]
    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    return result


def get_refusal_scores(model, tokenizer, instructions: List[str], refusal_toks: List[int],
                       fwd_pre_hooks=[], fwd_hooks=[], batch_size=BATCH_SIZE) -> Tensor:
    """Compute refusal scores for a list of instructions."""
    all_scores = torch.zeros(len(instructions), device=model.device)

    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i+batch_size]
        tokenized = tokenize_instructions(tokenizer, batch)

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            logits = model(
                input_ids=tokenized.input_ids.to(model.device),
                attention_mask=tokenized.attention_mask.to(model.device),
            ).logits

        all_scores[i:i+len(batch)] = refusal_score(logits, refusal_toks)

    return all_scores


def get_last_position_logits(model, tokenizer, instructions: List[str],
                             fwd_pre_hooks=[], fwd_hooks=[], batch_size=BATCH_SIZE) -> Tensor:
    """Get logits at last position for all instructions."""
    all_logits = None

    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i+batch_size]
        tokenized = tokenize_instructions(tokenizer, batch)

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            logits = model(
                input_ids=tokenized.input_ids.to(model.device),
                attention_mask=tokenized.attention_mask.to(model.device),
            ).logits

        batch_logits = logits[:, -1, :]
        if all_logits is None:
            all_logits = batch_logits
        else:
            all_logits = torch.cat((all_logits, batch_logits), dim=0)

    return all_logits


# ============================================================
# Main Pipeline
# ============================================================

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------
    # Step 1: Load model
    # ----------------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading model...")
    print("=" * 60)
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  Model loaded: {n_layers} layers, d_model={d_model}")
    print(f"  Device: {model.device}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Determine refusal tokens
    # Paper uses tokens for "I" and "As" — common refusal starters
    refusal_toks = []
    for word in ["I", "As", "Sorry"]:
        toks = tokenizer.encode(word, add_special_tokens=False)
        if toks:
            refusal_toks.append(toks[0])
    refusal_toks = list(set(refusal_toks))
    print(f"  Refusal tokens: {refusal_toks} → {[tokenizer.decode([t]) for t in refusal_toks]}")

    # Suppress <think> token for refusal scoring (reasoning model)
    global THINK_TOKEN_ID
    think_toks = tokenizer.encode("<think>", add_special_tokens=False)
    if think_toks:
        THINK_TOKEN_ID = think_toks[0]
        print(f"  Think token: {THINK_TOKEN_ID} (will be suppressed in refusal scoring)")

    # Determine EOI tokens (end-of-instruction, before assistant response)
    eoi_suffix = CHAT_TEMPLATE.split("{instruction}")[-1]
    eoi_toks = tokenizer.encode(eoi_suffix, add_special_tokens=False)
    n_eoi = len(eoi_toks)
    print(f"  EOI tokens ({n_eoi}): {eoi_toks} → {tokenizer.decode(eoi_toks)!r}")

    # Module references for Qwen 3.5 hybrid architecture
    # Layers alternate: linear_attn (GatedDeltaNet) and self_attn (full attention)
    # Full attn layers (every 4th: 3,7,11,...,63) have self_attn with o_proj
    # Linear attn layers have linear_attn with out_proj
    block_modules = list(model.model.layers)
    attn_modules = []
    for layer in model.model.layers:
        if hasattr(layer, 'self_attn'):
            attn_modules.append(layer.self_attn)
        elif hasattr(layer, 'linear_attn'):
            attn_modules.append(layer.linear_attn)
        else:
            raise ValueError(f"Layer {layer} has no attn module!")
    mlp_modules = [layer.mlp for layer in model.model.layers]

    full_attn_indices = [i for i, l in enumerate(model.model.layers) if hasattr(l, 'self_attn')]
    linear_attn_indices = [i for i, l in enumerate(model.model.layers) if hasattr(l, 'linear_attn')]
    print(f"  Full attention layers ({len(full_attn_indices)}): {full_attn_indices}")
    print(f"  Linear attention layers ({len(linear_attn_indices)}): {linear_attn_indices}")

    # ----------------------------------------------------------
    # Step 2: Load and sample datasets
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Loading datasets...")
    print("=" * 60)

    with open(os.path.join(DATA_DIR, "harmful_train.json")) as f:
        harmful_train_all = json.load(f)
    with open(os.path.join(DATA_DIR, "harmless_train.json")) as f:
        harmless_train_all = json.load(f)
    with open(os.path.join(DATA_DIR, "harmful_val.json")) as f:
        harmful_val_all = json.load(f)
    with open(os.path.join(DATA_DIR, "harmless_val.json")) as f:
        harmless_val_all = json.load(f)

    # Extract instruction text
    def get_instructions(data):
        if isinstance(data[0], dict):
            return [d["instruction"] for d in data]
        return data

    harmful_train_all = get_instructions(harmful_train_all)
    harmless_train_all = get_instructions(harmless_train_all)
    harmful_val_all = get_instructions(harmful_val_all)
    harmless_val_all = get_instructions(harmless_val_all)

    # Sample
    random.shuffle(harmful_train_all)
    random.shuffle(harmless_train_all)
    random.shuffle(harmful_val_all)
    random.shuffle(harmless_val_all)

    harmful_train_pool = harmful_train_all[:min(N_TRAIN * 2, len(harmful_train_all))]
    harmless_train_pool = harmless_train_all[:min(N_TRAIN * 2, len(harmless_train_all))]
    harmful_val_pool = harmful_val_all[:min(N_VAL * 2, len(harmful_val_all))]
    harmless_val_pool = harmless_val_all[:min(N_VAL * 2, len(harmless_val_all))]

    print(f"  Harmful train pool: {len(harmful_train_pool)}")
    print(f"  Harmless train pool: {len(harmless_train_pool)}")
    print(f"  Harmful val pool: {len(harmful_val_pool)}")
    print(f"  Harmless val pool: {len(harmless_val_pool)}")

    # ----------------------------------------------------------
    # Step 3: Filter prompts
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Filtering prompts by refusal score...")
    print("=" * 60)
    t0 = time.time()

    # Compute refusal scores for all pools
    print("  Computing refusal scores for harmful train pool...")
    harmful_train_scores = get_refusal_scores(model, tokenizer, harmful_train_pool, refusal_toks)
    print(f"    Mean score: {harmful_train_scores.mean():.4f}, refusing: {(harmful_train_scores > 0).sum()}/{len(harmful_train_pool)}")

    print("  Computing refusal scores for harmless train pool...")
    harmless_train_scores = get_refusal_scores(model, tokenizer, harmless_train_pool, refusal_toks)
    print(f"    Mean score: {harmless_train_scores.mean():.4f}, not refusing: {(harmless_train_scores < 0).sum()}/{len(harmless_train_pool)}")

    print("  Computing refusal scores for harmful val pool...")
    harmful_val_scores = get_refusal_scores(model, tokenizer, harmful_val_pool, refusal_toks)
    print(f"    Mean score: {harmful_val_scores.mean():.4f}, refusing: {(harmful_val_scores > 0).sum()}/{len(harmful_val_pool)}")

    print("  Computing refusal scores for harmless val pool...")
    harmless_val_scores = get_refusal_scores(model, tokenizer, harmless_val_pool, refusal_toks)
    print(f"    Mean score: {harmless_val_scores.mean():.4f}, not refusing: {(harmless_val_scores < 0).sum()}/{len(harmless_val_pool)}")

    # Filter: keep harmful that refuse (score > 0), harmless that don't (score < 0)
    harmful_train = [p for p, s in zip(harmful_train_pool, harmful_train_scores) if s.item() > 0][:N_TRAIN]
    harmless_train = [p for p, s in zip(harmless_train_pool, harmless_train_scores) if s.item() < 0][:N_TRAIN]
    harmful_val = [p for p, s in zip(harmful_val_pool, harmful_val_scores) if s.item() > 0][:N_VAL]
    harmless_val = [p for p, s in zip(harmless_val_pool, harmless_val_scores) if s.item() < 0][:N_VAL]

    print(f"\n  After filtering:")
    print(f"    Harmful train: {len(harmful_train)} (target {N_TRAIN})")
    print(f"    Harmless train: {len(harmless_train)} (target {N_TRAIN})")
    print(f"    Harmful val: {len(harmful_val)} (target {N_VAL})")
    print(f"    Harmless val: {len(harmless_val)} (target {N_VAL})")
    print(f"  Time: {time.time() - t0:.1f}s")

    assert len(harmful_train) > 0, "No harmful prompts passed filter!"
    assert len(harmless_train) > 0, "No harmless prompts passed filter!"
    assert len(harmful_val) > 0, "No harmful val prompts passed filter!"
    assert len(harmless_val) > 0, "No harmless val prompts passed filter!"

    # ----------------------------------------------------------
    # Step 4: Generate candidate directions (mean-diff)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Generating candidate directions...")
    print("=" * 60)
    t0 = time.time()

    positions = list(range(-n_eoi, 0))  # last N eoi positions
    n_positions = len(positions)
    print(f"  Positions: {positions} (last {n_positions} tokens)")

    def get_mean_activations(instructions: List[str]) -> Tensor:
        """Compute mean activations at eoi positions for each layer."""
        torch.cuda.empty_cache()
        n_samples = len(instructions)
        mean_acts = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

        fwd_pre_hooks = [
            (block_modules[layer],
             get_mean_activations_pre_hook(layer=layer, cache=mean_acts, n_samples=n_samples, positions=positions))
            for layer in range(n_layers)
        ]

        for i in tqdm(range(0, n_samples, BATCH_SIZE), desc="  Mean activations"):
            batch = instructions[i:i+BATCH_SIZE]
            tokenized = tokenize_instructions(tokenizer, batch)

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                model(
                    input_ids=tokenized.input_ids.to(model.device),
                    attention_mask=tokenized.attention_mask.to(model.device),
                )

        return mean_acts

    print(f"  Computing mean activations for {len(harmful_train)} harmful prompts...")
    mean_harmful = get_mean_activations(harmful_train)

    print(f"  Computing mean activations for {len(harmless_train)} harmless prompts...")
    mean_harmless = get_mean_activations(harmless_train)

    # Mean difference: the candidate refusal directions
    candidate_directions = mean_harmful - mean_harmless
    assert candidate_directions.shape == (n_positions, n_layers, d_model)
    assert not candidate_directions.isnan().any(), "NaN in candidate directions!"

    torch.save(candidate_directions, os.path.join(OUTPUT_DIR, "candidate_directions.pt"))
    print(f"  Candidate directions shape: {candidate_directions.shape}")
    print(f"  Norms range: [{candidate_directions.norm(dim=-1).min():.4f}, {candidate_directions.norm(dim=-1).max():.4f}]")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ----------------------------------------------------------
    # Step 5: Select best direction (3-point ablation evaluation)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Evaluating candidate directions...")
    print("=" * 60)
    t0 = time.time()

    # Baseline scores (no intervention)
    print("  Computing baseline refusal scores...")
    baseline_harmful_scores = get_refusal_scores(model, tokenizer, harmful_val, refusal_toks)
    baseline_harmless_scores = get_refusal_scores(model, tokenizer, harmless_val, refusal_toks)
    print(f"    Baseline harmful mean: {baseline_harmful_scores.mean():.4f}")
    print(f"    Baseline harmless mean: {baseline_harmless_scores.mean():.4f}")

    # Baseline harmless logits (for KL divergence)
    print("  Computing baseline harmless logits...")
    baseline_harmless_logits = get_last_position_logits(model, tokenizer, harmless_val)

    # Score matrices
    ablation_refusal_scores = torch.zeros((n_positions, n_layers), device=model.device, dtype=torch.float64)
    ablation_kl_div_scores = torch.zeros((n_positions, n_layers), device=model.device, dtype=torch.float64)
    steering_refusal_scores = torch.zeros((n_positions, n_layers), device=model.device, dtype=torch.float64)

    # 5a: KL divergence (ablation on harmless)
    print("\n  [5a] Computing KL divergence scores...")
    for source_pos_idx, source_pos in enumerate(range(-n_positions, 0)):
        for source_layer in tqdm(range(n_layers), desc=f"  KL div pos={source_pos}"):
            ablation_dir = candidate_directions[source_pos, source_layer]

            # 3-point ablation hooks
            fwd_pre_hooks = [
                (block_modules[l], get_direction_ablation_input_pre_hook(direction=ablation_dir))
                for l in range(n_layers)
            ]
            fwd_hooks = [
                (attn_modules[l], get_direction_ablation_output_hook(direction=ablation_dir))
                for l in range(n_layers)
            ]
            fwd_hooks += [
                (mlp_modules[l], get_direction_ablation_output_hook(direction=ablation_dir))
                for l in range(n_layers)
            ]

            intervention_logits = get_last_position_logits(
                model, tokenizer, harmless_val,
                fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks
            )

            ablation_kl_div_scores[source_pos, source_layer] = kl_div_fn(
                baseline_harmless_logits, intervention_logits
            ).mean(dim=0).item()

    # 5b: Ablation refusal score (ablation on harmful)
    print("\n  [5b] Computing ablation refusal scores...")
    for source_pos_idx, source_pos in enumerate(range(-n_positions, 0)):
        for source_layer in tqdm(range(n_layers), desc=f"  Ablation pos={source_pos}"):
            ablation_dir = candidate_directions[source_pos, source_layer]

            fwd_pre_hooks = [
                (block_modules[l], get_direction_ablation_input_pre_hook(direction=ablation_dir))
                for l in range(n_layers)
            ]
            fwd_hooks = [
                (attn_modules[l], get_direction_ablation_output_hook(direction=ablation_dir))
                for l in range(n_layers)
            ]
            fwd_hooks += [
                (mlp_modules[l], get_direction_ablation_output_hook(direction=ablation_dir))
                for l in range(n_layers)
            ]

            scores = get_refusal_scores(
                model, tokenizer, harmful_val, refusal_toks,
                fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks
            )
            ablation_refusal_scores[source_pos, source_layer] = scores.mean().item()

    # 5c: Steering score (activation addition on harmless)
    print("\n  [5c] Computing steering scores...")
    for source_pos_idx, source_pos in enumerate(range(-n_positions, 0)):
        for source_layer in tqdm(range(n_layers), desc=f"  Steering pos={source_pos}"):
            refusal_vector = candidate_directions[source_pos, source_layer]

            # Add direction at the candidate's layer only
            fwd_pre_hooks = [
                (block_modules[source_layer],
                 get_activation_addition_input_pre_hook(vector=refusal_vector, coeff=1.0))
            ]
            fwd_hooks = []

            scores = get_refusal_scores(
                model, tokenizer, harmless_val, refusal_toks,
                fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks
            )
            steering_refusal_scores[source_pos, source_layer] = scores.mean().item()

    print(f"\n  Evaluation time: {time.time() - t0:.1f}s")

    # Save raw scores
    torch.save({
        'ablation_refusal': ablation_refusal_scores,
        'ablation_kl_div': ablation_kl_div_scores,
        'steering_refusal': steering_refusal_scores,
        'baseline_harmful_mean': baseline_harmful_scores.mean().item(),
        'baseline_harmless_mean': baseline_harmless_scores.mean().item(),
    }, os.path.join(OUTPUT_DIR, "raw_scores.pt"))

    # ----------------------------------------------------------
    # Step 6: Filter and select best direction
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Selecting best direction...")
    print("=" * 60)

    all_scores_json = []
    filtered_scores = []
    filtered_json = []

    for source_pos in range(-n_positions, 0):
        for source_layer in range(n_layers):
            r_score = ablation_refusal_scores[source_pos, source_layer].item()
            s_score = steering_refusal_scores[source_pos, source_layer].item()
            k_score = ablation_kl_div_scores[source_pos, source_layer].item()

            entry = {
                'position': source_pos,
                'layer': source_layer,
                'refusal_score': r_score,
                'steering_score': s_score,
                'kl_div_score': k_score,
            }
            all_scores_json.append(entry)

            # Filter criteria
            if math.isnan(r_score) or math.isnan(s_score) or math.isnan(k_score):
                continue
            if source_layer >= int(n_layers * (1.0 - PRUNE_LAYER_PERCENTAGE)):
                continue
            if k_score > KL_THRESHOLD:
                continue
            if s_score < INDUCE_REFUSAL_THRESHOLD:
                continue

            # Sort by lowest refusal score (best at removing refusal)
            filtered_scores.append((-r_score, source_pos, source_layer))
            filtered_json.append(entry)

    # Save all scores
    with open(os.path.join(OUTPUT_DIR, "all_scores.json"), 'w') as f:
        json.dump(all_scores_json, f, indent=2)

    # Save filtered scores sorted
    filtered_json = sorted(filtered_json, key=lambda x: x['refusal_score'])
    with open(os.path.join(OUTPUT_DIR, "filtered_scores.json"), 'w') as f:
        json.dump(filtered_json, f, indent=2)

    print(f"  Total candidates: {n_positions * n_layers}")
    print(f"  After filtering: {len(filtered_scores)}")

    if len(filtered_scores) == 0:
        print("\n  WARNING: All candidates filtered! Relaxing criteria...")
        # Try without KL threshold
        for source_pos in range(-n_positions, 0):
            for source_layer in range(n_layers):
                r_score = ablation_refusal_scores[source_pos, source_layer].item()
                s_score = steering_refusal_scores[source_pos, source_layer].item()
                k_score = ablation_kl_div_scores[source_pos, source_layer].item()
                if math.isnan(r_score) or math.isnan(s_score) or math.isnan(k_score):
                    continue
                if source_layer >= int(n_layers * (1.0 - PRUNE_LAYER_PERCENTAGE)):
                    continue
                filtered_scores.append((-r_score, source_pos, source_layer))
        print(f"  After relaxed filtering: {len(filtered_scores)}")

    assert len(filtered_scores) > 0, "No candidates survived filtering!"

    # Sort descending by score (highest = best refusal removal)
    filtered_scores.sort(key=lambda x: x[0], reverse=True)

    _, best_pos, best_layer = filtered_scores[0]
    best_direction = candidate_directions[best_pos, best_layer].float()

    print(f"\n  SELECTED DIRECTION:")
    print(f"    Position: {best_pos}")
    print(f"    Layer: {best_layer}")
    print(f"    Refusal score: {ablation_refusal_scores[best_pos, best_layer]:.4f} (baseline: {baseline_harmful_scores.mean():.4f})")
    print(f"    Steering score: {steering_refusal_scores[best_pos, best_layer]:.4f} (baseline: {baseline_harmless_scores.mean():.4f})")
    print(f"    KL divergence: {ablation_kl_div_scores[best_pos, best_layer]:.4f}")
    print(f"    Direction norm: {best_direction.norm():.4f}")

    # Print top-5 candidates
    print(f"\n  Top-5 candidates:")
    for i, (score, pos, layer) in enumerate(filtered_scores[:5]):
        r = ablation_refusal_scores[pos, layer].item()
        s = steering_refusal_scores[pos, layer].item()
        k = ablation_kl_div_scores[pos, layer].item()
        print(f"    #{i+1}: pos={pos}, layer={layer}, refusal={r:.4f}, steering={s:.4f}, KL={k:.4f}")

    # ----------------------------------------------------------
    # Step 7: Save direction
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 7: Saving direction...")
    print("=" * 60)

    # Normalize direction
    direction_normalized = best_direction / best_direction.norm()

    # Save in SGLang-compatible format: [5120] tensor
    torch.save(direction_normalized, os.path.join(OUTPUT_DIR, "direction.pt"))
    print(f"  Saved: {OUTPUT_DIR}/direction.pt (shape {direction_normalized.shape})")

    # Also save unnormalized for analysis
    torch.save(best_direction, os.path.join(OUTPUT_DIR, "direction_unnormalized.pt"))

    # Save metadata
    metadata = {
        'position': best_pos,
        'layer': best_layer,
        'refusal_score': ablation_refusal_scores[best_pos, best_layer].item(),
        'steering_score': steering_refusal_scores[best_pos, best_layer].item(),
        'kl_div_score': ablation_kl_div_scores[best_pos, best_layer].item(),
        'direction_norm': best_direction.norm().item(),
        'baseline_harmful_refusal': baseline_harmful_scores.mean().item(),
        'baseline_harmless_refusal': baseline_harmless_scores.mean().item(),
        'n_train_harmful': len(harmful_train),
        'n_train_harmless': len(harmless_train),
        'n_val_harmful': len(harmful_val),
        'n_val_harmless': len(harmless_val),
        'model_path': MODEL_PATH,
        'n_layers': n_layers,
        'd_model': d_model,
        'eoi_toks': eoi_toks,
        'refusal_toks': refusal_toks,
    }
    with open(os.path.join(OUTPUT_DIR, "direction_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR}/direction_metadata.json")

    # Compare to wdiff if available
    wdiff_path = "/tmp/wdiff_direction_global.pt"
    if os.path.exists(wdiff_path):
        wdiff = torch.load(wdiff_path, map_location=model.device, weights_only=True).float()
        wdiff_norm = wdiff / wdiff.norm()
        cos_sim = (direction_normalized.to(wdiff_norm.device) @ wdiff_norm).item()
        print(f"\n  Cosine similarity to wdiff: {cos_sim:.4f}")
        metadata['cosine_to_wdiff'] = cos_sim
        with open(os.path.join(OUTPUT_DIR, "direction_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print("DONE! Direction extracted successfully.")
    print(f"{'=' * 60}")
    print(f"\nTo deploy:")
    print(f"  cp {OUTPUT_DIR}/direction.pt /tmp/arditi_direction.pt")
    print(f"  # Then restart SGLang with --abliteration-vector-path /tmp/arditi_direction.pt")


if __name__ == "__main__":
    main()
