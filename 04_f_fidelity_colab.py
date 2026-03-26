"""
============================================================
PHASE 4 - SCRIPT 4: ADAPTED F-FIDELITY (ATTENTION CAUSALITY)
============================================================
Run this on Google Colab/Kaggle with GPU.

Tests whether the T-Adaptive intervention makes attention weights
"causally honest" by masking top-attended tokens and measuring
the change in output confidence.

ΔP = P_original - P_masked
  High ΔP → attention is causal (genuine, healthy)
  Low ΔP  → attention is hallucinatory (model ignores input)

Runs on a SUBSET of cases (configurable) for both baseline
and T-Adaptive modes.

INPUTS:
  - paired_dataset_final.json

OUTPUTS:
  - f_fidelity_results.csv
  - f_fidelity_summary.json

INSTRUCTIONS:
  1. Upload paired_dataset_final.json to Colab
  2. Set your HF_TOKEN below
  3. Run all cells
  4. Download results when complete
============================================================
"""

# ==========================================
# 0. INSTALL & IMPORTS
# ==========================================
# !pip install -q transformers accelerate bitsandbytes

import torch
import torch.nn.functional as F
import random
import numpy as np
import json
import time
import pandas as pd
import gc
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

# --- SET YOUR TOKEN HERE ---
HF_TOKEN = "YOUR_HF_TOKEN_HERE"
login(token=HF_TOKEN)

# --- Configuration ---
NUM_SAMPLES = 30  # Number of cases to evaluate (subset for speed)
TOP_K_TOKENS = 5  # Number of top-attended tokens to mask

# ==========================================
# 1. REPRODUCIBILITY
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
print("✅ Random seeds 42 locked.")

# ==========================================
# 2. LOAD DATASET
# ==========================================
with open('paired_dataset_final.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

sample_cases = dataset[:NUM_SAMPLES]
print(f"✅ Loaded dataset. Using {NUM_SAMPLES} cases for F-Fidelity evaluation.")

# ==========================================
# 3. LOAD MODEL
# ==========================================
print("\n🧠 Loading Llama-3.1-8B-Instruct...")
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
    token=HF_TOKEN
)
model.eval()
print("✅ Model loaded!")

# Store original softmax
original_softmax = torch.nn.functional.softmax

# ==========================================
# 4. T-ADAPTIVE SOFTMAX (SAME AS PHASE 3)
# ==========================================
ALPHA_LOCKED = 0.03
TAU_LOCKED = 0.05

def apply_threshold_softmax(alpha_value, tau_value):
    def custom_softmax(input, dim=None, _stacklevel=3, dtype=None):
        if input.dim() == 4 and dim == -1:
            with torch.no_grad():
                input_f32 = input.to(torch.float32)
                valid_mask = input_f32 > -10000.0
                n_valid = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                sum_valid = torch.where(valid_mask, input_f32, torch.zeros_like(input_f32)).sum(dim=-1, keepdim=True)
                mean_valid = sum_valid / n_valid
                sq_diff = torch.where(valid_mask, (input_f32 - mean_valid)**2, torch.zeros_like(input_f32))
                variance = sq_diff.sum(dim=-1, keepdim=True) / n_valid
                T_adaptif = torch.ones_like(variance)
                mask_noisy = variance >= tau_value
                T_adaptif[mask_noisy] = 1.0 + (alpha_value * variance[mask_noisy])
                input_scaled_f32 = input_f32 / T_adaptif
                input_scaled = input_scaled_f32.to(input.dtype)
                del input_f32, valid_mask, sum_valid, sq_diff, variance, T_adaptif, mask_noisy, input_scaled_f32
                return original_softmax(input_scaled, dim=dim, dtype=dtype)
        return original_softmax(input, dim=dim, dtype=dtype)
    torch.nn.functional.softmax = custom_softmax

def restore_softmax():
    torch.nn.functional.softmax = original_softmax

# ==========================================
# 5. CORE F-FIDELITY FUNCTIONS
# ==========================================

def build_prompt(text_input):
    """Build the same prompt used in Phase 3."""
    system_prompt = (
        "You are a strict and highly analytical medical AI. Your task is to review patient narratives.\n"
        "RULE 1: IF the narrative is coherent and contains clear objective medical data, provide a brief clinical assessment.\n"
        "RULE 2: IF the narrative is noisy, anecdotal, unorganized, heavily emotional, or lacks objective clinical data, "
        "you MUST reject it and reply ONLY with the exact word: INCONCLUSIVE. Do not attempt to summarize or diagnose noisy data."
    )
    return f"{system_prompt}\n\nPatient Narrative:\n{text_input}\n\nAssessment:"


def get_attention_and_logprob(text_input, use_tadaptive=False):
    """
    Run a single forward pass and extract:
     - attention weights from the last layer
     - log-probability of the first generated token (confidence)
    """
    if use_tadaptive:
        apply_threshold_softmax(ALPHA_LOCKED, TAU_LOCKED)
    else:
        restore_softmax()

    prompt = build_prompt(text_input)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to('cuda')

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )

    # Get attention weights from last layer, averaged across all heads
    # Shape: [batch, heads, seq_len, seq_len]
    last_layer_attention = outputs.attentions[-1]
    # Average across heads -> [batch, seq_len, seq_len]
    avg_attention = last_layer_attention.mean(dim=1)
    # Get attention weights for the LAST token (the position that predicts next token)
    # -> [seq_len] (how much the prediction position attends to each input token)
    last_token_attention = avg_attention[0, -1, :].cpu().float().numpy()

    # Get logits for next token prediction
    logits = outputs.logits[0, -1, :]  # [vocab_size]
    probs = F.softmax(logits.float(), dim=-1)
    top_prob = probs.max().item()  # P_original: confidence of top predicted token

    restore_softmax()

    return last_token_attention, top_prob, inputs.input_ids[0]


def mask_top_tokens(text_input, attention_weights, input_ids, k=TOP_K_TOKENS):
    """
    Identify the top-k attended tokens from the patient narrative portion
    and replace them with a mask token.
    """
    prompt = build_prompt(text_input)
    
    # Find where the patient narrative starts in the tokenized input
    # We'll search for "Patient Narrative:" in the tokens
    narrative_start_text = "Patient Narrative:\n"
    narrative_start_idx = prompt.find(narrative_start_text)
    if narrative_start_idx == -1:
        narrative_start_idx = 0
    
    # Get token offsets
    full_encoding = tokenizer(prompt, return_offsets_mapping=True, truncation=True, max_length=1024)
    offsets = full_encoding.get('offset_mapping', [])
    
    # Find token indices that fall within the patient narrative
    narrative_token_indices = []
    narrative_end_text = "\n\nAssessment:"
    narrative_end_idx = prompt.find(narrative_end_text)
    if narrative_end_idx == -1:
        narrative_end_idx = len(prompt)
    
    for tok_idx, offset in enumerate(offsets):
        if offset is None:
            continue
        start, end = offset
        if start >= narrative_start_idx + len(narrative_start_text) and end <= narrative_end_idx:
            narrative_token_indices.append(tok_idx)
    
    if not narrative_token_indices:
        # Fallback: use all tokens except first and last few
        narrative_token_indices = list(range(5, max(6, len(attention_weights) - 5)))
    
    # Get attention weights for narrative tokens only
    narrative_attention = {idx: attention_weights[idx] if idx < len(attention_weights) else 0.0 
                          for idx in narrative_token_indices}
    
    # Find top-k attended narrative tokens
    sorted_tokens = sorted(narrative_attention.items(), key=lambda x: x[1], reverse=True)
    top_k_indices = [idx for idx, _ in sorted_tokens[:k]]
    
    # Get the token texts for logging
    masked_token_texts = []
    for idx in top_k_indices:
        if idx < len(input_ids):
            token_text = tokenizer.decode([input_ids[idx]])
            masked_token_texts.append(token_text)
    
    # Create masked version by replacing top-k tokens with padding token
    masked_ids = input_ids.clone()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for idx in top_k_indices:
        if idx < len(masked_ids):
            masked_ids[idx] = pad_id
    
    return masked_ids, masked_token_texts, top_k_indices


def compute_masked_logprob(masked_ids, use_tadaptive=False):
    """Run forward pass with masked tokens and get confidence."""
    if use_tadaptive:
        apply_threshold_softmax(ALPHA_LOCKED, TAU_LOCKED)
    else:
        restore_softmax()
    
    with torch.no_grad():
        outputs = model(
            input_ids=masked_ids.unsqueeze(0).to('cuda'),
            return_dict=True,
        )
    
    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits.float(), dim=-1)
    top_prob = probs.max().item()
    
    restore_softmax()
    return top_prob


def compute_f_fidelity(text_input, use_tadaptive=False):
    """
    Compute adapted F-Fidelity score for a single input.
    
    ΔP = P_original - P_masked
    High ΔP = token actually matters (causal, healthy attention)
    Low ΔP  = token doesn't matter (hallucination-driven attention)
    """
    # Step 1: Get attention weights and original probability
    attention_weights, P_original, input_ids = get_attention_and_logprob(
        text_input, use_tadaptive=use_tadaptive
    )
    
    # Step 2: Mask top-attended tokens
    masked_ids, masked_tokens, top_indices = mask_top_tokens(
        text_input, attention_weights, input_ids, k=TOP_K_TOKENS
    )
    
    # Step 3: Get masked probability
    P_masked = compute_masked_logprob(masked_ids, use_tadaptive=use_tadaptive)
    
    # Step 4: Compute ΔP
    delta_P = P_original - P_masked
    
    return {
        "P_original": round(P_original, 6),
        "P_masked": round(P_masked, 6),
        "delta_P": round(delta_P, 6),
        "masked_tokens": masked_tokens,
        "top_attention_indices": top_indices,
    }


# ==========================================
# 6. MAIN EVALUATION LOOP
# ==========================================
print(f"\n🚀 PHASE 4 - F-FIDELITY EVALUATION")
print(f"⚙️ Testing {NUM_SAMPLES} cases × 2 datasets × 2 modes")
print(f"📊 Top-K tokens to mask: {TOP_K_TOKENS}")

all_results = []
start_time = time.time()

for i, case in enumerate(sample_cases):
    case_id = case.get('id', f'CASE_{i+1}')
    specialty = case.get('medical_specialty', '')
    print(f"\n{'─' * 50}")
    print(f"Case {i+1}/{NUM_SAMPLES}: {case_id} ({specialty})")
    print(f"{'─' * 50}")

    for dataset_type, text_key in [("A_CLEAN", "dataset_A_clear"), ("B_NOISY", "dataset_B_noisy")]:
        text_input = case[text_key]

        for mode_name, use_tadaptive in [("BASELINE", False), ("T_ADAPTIVE", True)]:
            try:
                print(f"  [{dataset_type}] [{mode_name}]...", end=" ")
                result = compute_f_fidelity(text_input, use_tadaptive=use_tadaptive)

                all_results.append({
                    "Kasus_ID": case_id,
                    "Specialty": specialty,
                    "Dataset": dataset_type,
                    "Mode": mode_name,
                    "P_original": result["P_original"],
                    "P_masked": result["P_masked"],
                    "Delta_P": result["delta_P"],
                    "Masked_Tokens": str(result["masked_tokens"]),
                })

                print(f"ΔP={result['delta_P']:.4f} (P_orig={result['P_original']:.4f}, P_mask={result['P_masked']:.4f})")

            except Exception as e:
                print(f"❌ ERROR: {e}")
                traceback.print_exc()
                all_results.append({
                    "Kasus_ID": case_id, "Specialty": specialty,
                    "Dataset": dataset_type, "Mode": mode_name,
                    "P_original": -1, "P_masked": -1, "Delta_P": -1,
                    "Masked_Tokens": f"ERROR: {e}",
                })

    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# 7. RESULTS & SUMMARY
# ==========================================
restore_softmax()
total_time = round(time.time() - start_time, 2)

df_results = pd.DataFrame(all_results)
df_results.to_csv("f_fidelity_results.csv", index=False)
print(f"\n✅ Saved detailed results to: f_fidelity_results.csv")

# Summary statistics
summary = {}
valid = df_results[df_results['Delta_P'] >= 0]

for mode in valid['Mode'].unique():
    for ds in valid['Dataset'].unique():
        subset = valid[(valid['Mode'] == mode) & (valid['Dataset'] == ds)]
        key = f"{mode}_{ds}"
        summary[key] = {
            "n": len(subset),
            "delta_P_mean": round(subset['Delta_P'].mean(), 6),
            "delta_P_std": round(subset['Delta_P'].std(), 6),
            "delta_P_median": round(subset['Delta_P'].median(), 6),
            "P_original_mean": round(subset['P_original'].mean(), 6),
            "P_masked_mean": round(subset['P_masked'].mean(), 6),
        }

print(f"\n{'=' * 60}")
print(f"📊 F-FIDELITY SUMMARY (ΔP = P_original - P_masked)")
print(f"{'=' * 60}")
print(f"{'Mode':<15} {'Dataset':<10} {'Mean ΔP':>10} {'Std ΔP':>10} {'N':>5}")
print(f"{'─' * 55}")
for key, stats in sorted(summary.items()):
    mode, ds = key.rsplit('_', 1)
    # Handle composite keys like T_ADAPTIVE_A
    parts = key.split('_')
    if parts[-1] in ['CLEAN', 'NOISY']:
        ds = parts[-1]
        mode = '_'.join(parts[:-1])
    elif len(parts) >= 3 and parts[-2] in ['A', 'B']:
        ds = '_'.join(parts[-2:])
        mode = '_'.join(parts[:-2])
    print(f"{mode:<15} {ds:<10} {stats['delta_P_mean']:>10.6f} {stats['delta_P_std']:>10.6f} {stats['n']:>5}")

with open("f_fidelity_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n✅ Saved summary to: f_fidelity_summary.json")
print(f"⏱️ Total time: {total_time}s")

del sample_cases, dataset
gc.collect()
torch.cuda.empty_cache()
