"""
============================================================
PHASE 4 - SCRIPT 3: RAGAS FAITHFULNESS EVALUATION (GROQ VERSION)
============================================================
Evaluates Faithfulness using Groq (meta-llama/llama-4-scout-17b-16e-instruct)
Blazing fast execution with 30 RPM limit handling.
(FITUR: RESUME DARI KASUS 147 & AUTO-SAVE KE CSV)
"""

import json
import os
import time
import pandas as pd
from openai import OpenAI

# --- SET YOUR GROQ API KEY HERE ---
# (Catatan: Hati-hati jangan bagikan API Key ini ke tempat publik ya!)
GROQ_API_KEY = "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"

# --- Configuration ---
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DELAY_BETWEEN_CALLS = 2.1  # Aman untuk limit 30 RPM Groq
MAX_RETRIES = 3
CSV_FILENAME = "ragas_faithfulness_results.csv"

# KITA MULAI DARI KASUS 147 (Karena indeks Python mulai dari 0, maka nilainya 146)
START_INDEX = 146  

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

def get_faithfulness_score(source_text, generated_text):
    # Jika model sudah menolak (Inconclusive), faithfulness otomatis 1.0 (100%)
    if "inconclusive" in generated_text.lower()[:50]:
        return 1.0

    # GABUNGAN: Ekstrak dan Verifikasi dalam 1 kali panggil (Hemat 50% Token)
    prompt_combined = f"""
    You are a strict medical evaluator.
    
    Source Medical Document:
    {source_text}
    
    Generated Assessment:
    {generated_text}
    
    Task:
    1. Extract all specific medical claims (diagnoses, symptoms, treatments) from the Generated Assessment.
    2. Check if EACH claim is supported by the Source Medical Document.
    
    Return ONLY a JSON object exactly in this format:
    {{
      "supported_claims": ["claim 1", "claim 2"],
      "unsupported_claims": ["claim 3"]
    }}
    """
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "Always return valid JSON."},
                    {"role": "user", "content": prompt_combined}
                ],
                temperature=0.0,
                max_tokens=400, # Membatasi agar model tidak ngoceh kepanjangan
                response_format={"type": "json_object"}
            )
            result_json = json.loads(response.choices[0].message.content)
            
            supported = result_json.get("supported_claims", [])
            unsupported = result_json.get("unsupported_claims", [])
            total_claims = len(supported) + len(unsupported)
            
            # Hitung skor (jumlah yg di-support dibagi total klaim)
            score = len(supported) / total_claims if total_claims > 0 else 0.0
            return score
            
        except Exception as e:
            print(f"      ⚠ Attempt {attempt+1} failed: {e}")
            time.sleep(DELAY_BETWEEN_CALLS * 2)
            
    return 0.0

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"🚀 Melanjutkan Evaluasi Faithfulness menggunakan Groq ({GROQ_MODEL})")
    print(f"📂 Mode Auto-Save Aktif. Menyambung data ke: {CSV_FILENAME}")
    
    # Load Datasets
    with open("paired_dataset_final.json", "r") as f:
        dataset = json.load(f)
        
    df_baseline = pd.read_csv("hasil_baseline_llama.csv")
    df_tadaptive = pd.read_csv("hasil_evaluasi_fase4_llama.csv") # Sesuaikan nama file jika perlu
    
    # Looping dimulai dari START_INDEX (146) agar langsung ke Kasus 147
    for i, case in enumerate(dataset[START_INDEX:]):
        real_index = i + START_INDEX  # Memastikan kita mengambil baris yang benar di CSV Llama
        print(f"\n⏳ Memproses Kasus {real_index + 1}/{len(dataset)}...")
        
        source_doc = case['dataset_A_clear']
        
        # Evaluasi Baseline (Llama Biasa)
        out_base_A = str(df_baseline.iloc[real_index]['Output_A_Llama'])
        score_base = get_faithfulness_score(source_doc, out_base_A)
        print(f"   [Baseline] Score: {score_base:.2f}")
        time.sleep(DELAY_BETWEEN_CALLS)
        
        # Evaluasi T-Adaptive (Llama Modifikasi)
        out_tadapt_A = str(df_tadaptive.iloc[real_index]['Output_A_Llama'])
        score_tadapt = get_faithfulness_score(source_doc, out_tadapt_A)
        print(f"   [T-Adaptive] Score: {score_tadapt:.2f}")
        time.sleep(DELAY_BETWEEN_CALLS)
        
        # SIMPAN LANGSUNG KE CSV SETIAP SELESAI 1 KASUS
        row_data = pd.DataFrame([{
            "Kasus_ID": case['id'],
            "Faithfulness_Baseline": score_base,
            "Faithfulness_TAdaptive": score_tadapt
        }])
        
        # mode='a' berarti "append" (tambahkan ke baris paling bawah)
        row_data.to_csv(CSV_FILENAME, mode='a', header=not os.path.exists(CSV_FILENAME), index=False)

    print("\n" + "="*50)
    print("🏁 EVALUASI SELESAI!")
    
    # Menghitung Rata-rata Total dengan membaca kembali seluruh isi file CSV (Kasus 1 - 200)
    if os.path.exists(CSV_FILENAME):
        final_df = pd.read_csv(CSV_FILENAME)
        print(f"Total Kasus Dievaluasi             : {len(final_df)}")
        print(f"Rata-rata Faithfulness Baseline    : {final_df['Faithfulness_Baseline'].mean():.2f}")
        print(f"Rata-rata Faithfulness T-Adaptive  : {final_df['Faithfulness_TAdaptive'].mean():.2f}")
    
    print(f"File '{CSV_FILENAME}' telah lengkap dan tersimpan!")
    print("="*50)