"""
============================================================
PHASE 4 - SCRIPT 1: BASELINE TEST (NO T-ADAPTIVE)
============================================================
Run this on Google Colab/Kaggle with GPU.
This runs VANILLA Llama-3.1-8B-Instruct on all 200 cases
WITHOUT any T-Adaptive Softmax intervention.

Produces: hasil_baseline_llama.csv
  Columns: Kasus_ID, Output_A_Llama, Output_B_Llama, 
           Is_A_Diagnosis, Is_B_Inconclusive

INSTRUCTIONS:
1. Upload paired_dataset_final.json to your Colab session
2. Set your HF_TOKEN below
3. Run all cells
4. Download hasil_baseline_llama.csv when complete
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
HF_TOKEN = "put your API key here"
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
# 4. BASELINE GENERATION FUNCTION (NO INTERVENTION)
# ==========================================
def generate_diagnosis_baseline(text_input):
    """
    Baseline inference: Standard Llama with NO T-Adaptive intervention.
    Uses the same strict system prompt as the T-Adaptive test for fair comparison.
    """
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
# 5. RUN BASELINE ON ALL 200 CASES
# ==========================================
print(f"\n🚀 PHASE 4 - BASELINE TEST (NO T-ADAPTIVE)")
print(f"⚙️ Running baseline evaluation on {len(dataset)} cases...")
print(f"📊 Parameters: VANILLA Llama (No Softmax modification)")

score_A, score_B = 0, 0
start_time = time.time()
log_results = []

for i, case in enumerate(dataset):
    print(f"\n⏳ Processing Case {i+1}/{len(dataset)}...")

    try:
        # --- Scenario A (Clear/Clean input) ---
        output_A = generate_diagnosis_baseline(case['dataset_A_clear'])
        is_A_diagnosis = "inconclusive" not in output_A.lower()
        if is_A_diagnosis:
            score_A += 1

        # --- Scenario B (Noisy input) ---
        output_B = generate_diagnosis_baseline(case['dataset_B_noisy'])
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
# 6. RESULTS & EXPORT
# ==========================================
total_time = round(time.time() - start_time, 2)
accuracy_A = (score_A / len(dataset)) * 100
accuracy_B = (score_B / len(dataset)) * 100

print("\n" + "=" * 50)
print("🏁 BASELINE EVALUATION COMPLETE!")
print(f"📈 Accuracy A (Clean Data)  : {accuracy_A}% ({score_A}/{len(dataset)})")
print(f"📉 Accuracy B (Noisy Data)  : {accuracy_B}% ({score_B}/{len(dataset)})")
print(f"⏱️ Total Execution Time     : {total_time} seconds")
print("=" * 50)

df_results = pd.DataFrame(log_results)
df_results.to_csv("hasil_baseline_llama.csv", index=False)
print("💾 'hasil_baseline_llama.csv' saved!")

# Summary row
summary = {
    "test_type": "BASELINE (No T-Adaptive)",
    "accuracy_A_pct": accuracy_A,
    "accuracy_B_pct": accuracy_B,
    "score_A": score_A,
    "score_B": score_B,
    "total_cases": len(dataset),
    "total_time_seconds": total_time,
}
with open("summary_baseline.json", "w") as f:
    json.dump(summary, f, indent=2)
print("💾 'summary_baseline.json' saved!")

del dataset
gc.collect()
torch.cuda.empty_cache()
