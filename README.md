# temperature-adaptive-softmax

This repository contains the complete dataset, generated outputs, evaluation metrics, and Python source code required to replicate the findings presented in the paper "The Alignment Tax of Causal Attention: Mitigating LLM Hallucinations via Temperature-Adaptive Softmax on Noisy Medical Records".

The research investigates the phenomenon of "prior-overreliance" in Large Language Models (LLMs) when processing noisy, unstructured Electronic Health Records (EHRs). We introduce T-Adaptive Attention, a training-free, inference-time intervention that dynamically scales the Softmax temperature based on the variance of the model's internal value states ($\alpha = 0.03$, $\tau = 0.05$). The experiments were conducted using the Meta-Llama-3.1-8B-Instruct model in a 4-bit quantized configuration.

This repository provides:

1. The Paired Clinical Dataset: 200 medical cases from 10 diagnostic specialties, structured into Dataset A (Clean/Structured) and Dataset B (Noisy/Unstructured).

2. Evaluation Results (Black-Box & White-Box): Raw and processed data for Classification Accuracy, RAGAS Faithfulness (evaluated via Llama-4-Scout-17B), and F-Fidelity ($\Delta P$) causal attention metrics.

3. Source Code: Python scripts for implementing the dynamic Softmax monkey-patching, running the LLM-as-a-Judge pipeline, extracting attention weights in eager mode, and generating the statistical analyses (McNemar's, Wilcoxon, Mann-Whitney U).
