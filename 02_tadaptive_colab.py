"""
============================================================
PHASE 4 - SCRIPT 2: T-ADAPTIVE RE-RUN WITH FULL OUTPUT CAPTURE
============================================================
Run this on Google Colab/Kaggle with GPU.
This runs T-Adaptive Llama (α=0.03, τ=0.05) on all 200 cases
and saves the FULL generated text for RAGAS evaluation.

Produces: hasil_tadaptive_llama.csv
  Columns: Kasus_ID, Specialty, Output_A_Llama, Output_B_Llama,
           Is_A_Diagnosis, Is_B_Inconclusive

INSTRUCTIONS:
1. Upload paired_dataset_final.json to your Colab session
2. Set your HF_TOKEN below
3. Run all cells
4. Download hasil_tadaptive_llama.csv when complete

NOTE: If you already have the full-test CSV with output text columns,
you can skip this script and use that CSV directly.
============================================================
"""

# ==========================================
# 0. INSTALL & CONFIG
# ==========================================
# !pip install -q transformers accelerate bitsandbytes sentence-transformers chromadb

import torch
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

# ==========================================
# 1. REPRODUCIBILITY (SEED = 42)
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
print(f"✅ Loaded {len(dataset)} cases from paired_dataset_final.json")

# ==========================================
# 3. LOAD LLAMA 3.1 8B (4-BIT QUANTIZED)
# ==========================================
print("\n🧠 Loading Llama-3.1-8B-Instruct to GPU (2-3 minutes)...")
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
print("✅ Llama 3.1 8B loaded to GPU!")

# ==========================================
# 4. T-ADAPTIVE SOFTMAX OVERRIDE
# ==========================================
original_softmax = torch.nn.functional.softmax

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

# ==========================================
# 5. GENERATION FUNCTION
# ==========================================
def generate_diagnosis(text_input):
    system_prompt = (
        "You are a strict and highly analytical medical AI. Your task is to review patient narratives.\n"
        "RULE 1: IF the narrative is coherent and contains clear objective medical data, provide a brief clinical assessment.\n"
        "RULE 2: IF the narrative is noisy, anecdotal, unorganized, heavily emotional, or lacks objective clinical data, "
        "you MUST reject it and reply ONLY with the exact word: INCONCLUSIVE. Do not attempt to summarize or diagnose noisy data."
    )

    full_prompt = f"{system_prompt}\n\nPatient Narrative:\n{text_input}\n\nAssessment:"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024).to('cuda')

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return result

# ==========================================
# 6. RUN T-ADAPTIVE ON ALL 200 CASES
# ==========================================
print(f"\n🚀 PHASE 4 - T-ADAPTIVE TEST (α={ALPHA_LOCKED}, τ={TAU_LOCKED})")
print(f"⚙️ Running T-Adaptive evaluation on {len(dataset)} cases...")

apply_threshold_softmax(alpha_value=ALPHA_LOCKED, tau_value=TAU_LOCKED)

score_A, score_B = 0, 0
start_time = time.time()
log_results = []

for i, case in enumerate(dataset):
    print(f"\n⏳ Processing Case {i+1}/{len(dataset)}...")

    try:
        # --- Scenario A (Clear/Clean input) ---
        output_A = generate_diagnosis(case['dataset_A_clear'])
        is_A_diagnosis = "inconclusive" not in output_A.lower()
        if is_A_diagnosis:
            score_A += 1

        # --- Scenario B (Noisy input) ---
        output_B = generate_diagnosis(case['dataset_B_noisy'])
        is_B_inconclusive = "inconclusive" in output_B.lower()
        if is_B_inconclusive:
            score_B += 1

        log_results.append({
            "Kasus_ID": case.get('id', f"CASE_{i+1}"),
            "Specialty": case.get('medical_specialty', ''),
            "Output_A_Llama": output_A,
            "Output_B_Llama": output_B,
            "Is_A_Diagnosis": is_A_diagnosis,
            "Is_B_Inconclusive": is_B_inconclusive,
        })

        print(f"   ✅ Done. (Running Score -> A: {score_A}, B: {score_B})")

    except Exception as e:
        print(f"   ❌ ERROR on Case {i+1}: {e}")
        traceback.print_exc()
        log_results.append({
            "Kasus_ID": case.get('id', f"CASE_{i+1}"),
            "Specialty": case.get('medical_specialty', ''),
            "Output_A_Llama": f"ERROR: {e}",
            "Output_B_Llama": f"ERROR: {e}",
            "Is_A_Diagnosis": False,
            "Is_B_Inconclusive": False,
        })

    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# 7. RESULTS & EXPORT
# ==========================================
torch.nn.functional.softmax = original_softmax  # Restore original

total_time = round(time.time() - start_time, 2)
accuracy_A = (score_A / len(dataset)) * 100
accuracy_B = (score_B / len(dataset)) * 100

print("\n" + "=" * 50)
print("🏁 T-ADAPTIVE EVALUATION COMPLETE!")
print("✅ Llama softmax restored to factory standard.")
print(f"📈 Accuracy A (Clean Data)  : {accuracy_A}% ({score_A}/{len(dataset)})")
print(f"📉 Accuracy B (Noisy Data)  : {accuracy_B}% ({score_B}/{len(dataset)})")
print(f"⏱️ Total Execution Time     : {total_time} seconds")
print(f"📊 Parameters: α={ALPHA_LOCKED}, τ={TAU_LOCKED}")
print("=" * 50)

df_results = pd.DataFrame(log_results)
df_results.to_csv("hasil_tadaptive_llama.csv", index=False)
print("💾 'hasil_tadaptive_llama.csv' saved!")

summary = {
    "test_type": "T-ADAPTIVE",
    "alpha": ALPHA_LOCKED,
    "tau": TAU_LOCKED,
    "accuracy_A_pct": accuracy_A,
    "accuracy_B_pct": accuracy_B,
    "score_A": score_A,
    "score_B": score_B,
    "total_cases": len(dataset),
    "total_time_seconds": total_time,
}
with open("summary_tadaptive.json", "w") as f:
    json.dump(summary, f, indent=2)
print("💾 'summary_tadaptive.json' saved!")

del dataset
gc.collect()
torch.cuda.empty_cache()
