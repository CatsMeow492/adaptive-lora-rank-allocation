Step-by-Step Experimental Design

Goal: Validate whether joint adaptive LoRA-rank + mixed-precision quantization outperforms (a) fixed-rank LoRA, (b) adaptive-rank only, and (c) quantized LoRA only, under MacBook-class hardware.

⸻

0.  Prep Checklist

Item	Choice
Frameworks	🤗 Transformers 4.40+, PEFT 0.10+, bitsandbytes 0.43 (CPU-safe build)
Device	Apple-silicon MacBook (M-series) — use MPS for FP16 runs; 8-bit quant runs on CPU; 4-bit runs via tiny cloud GPU if local int4 kernels unavailable
Task	SST-2 sentiment (GLUE) and WikiText-2 LM (≈ 2 h each) → gives both classification & LM metrics
Model	bert-base-uncased (110 M) for SST-2, gpt-2-small (117 M) for LM
Metrics	SST-2 accuracy, WikiText-2 valid perplexity + training wall-clock + peak RAM


⸻

1.  Environment & Data

1.1 Create conda/venv; install libraries.
1.2 datasets load SST-2 train/valid (≈ 67 k tokens) and WikiText-2.
1.3 Tokenize with HF fast tokenizers; cache to disk.

⸻

2.  Baseline Configurations

ID	Quantization	LoRA rank pattern	Trainable %
B-FP	16-bit (no quant)	Fixed r = 8 all layers	~0.15%
B-Q4	4-bit NF4 (QLoRA)	Fixed r = 8	~0.15%
B-Ada	16-bit	AdaLoRA (init r = 12 ➜ target r = 4)	adaptive (~0.08–0.12%)


⸻

3.  Joint Adaptive + Precision Variants

ID	Quantization plan	Rank plan
Joint-1	All layers 4-bit	AdaLoRA as in B-Ada
Joint-2	Embedding, layer-0, layer-(L-1) 8-bit; rest 4-bit	AdaLoRA
Joint-3	Attention matrices 8-bit, FFN 4-bit	Manual rank pattern: attention r = 6, FFN r = 10

Rationale: Joint-2 tests “critical layers higher precision”; Joint-3 tests precision-rank complementarity.

⸻

4.  Implementation Details

4.1 Load & quantize:

config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model = AutoModel.from_pretrained(…, quantization_config=config)
model = prepare_model_for_kbit_training(model)  # PEFT helper

For Joint-2/3, set module.exq_quant_config.skip=True on chosen layers or reload them in 8-bit.

4.2 Attach LoRA / AdaLoRA:

lora_cfg = AdaLoraConfig(…init_r=12, target_r=4)  # or LoraConfig(rank=8)
model = get_peft_model(model, lora_cfg)

4.3 Training hyper-params:
	•	LR = 2 e-4, AdamW, batch = 8 (SST-2) / 4 (Wiki); 3 epochs.
	•	Gradient Accum = 2 on CPU to keep RAM ≤ 16 GB.
	•	Learning-rate warm-up 10 %; cosine decay.
	•	Clip grad = 1.0.
	•	Log loss, grad-norm every 50 steps.

⸻

5.  Run Matrix

Task	B-FP	B-Q4	B-Ada	Joint-1	Joint-2	Joint-3
SST-2	✓	✓	✓	✓	✓	✓
Wikitext-2	✓	✓	✓	✓	✓	✓

(6 configs × 2 tasks = 12 runs; each ≈ 0.5 – 1 h on Mac; schedule overnight if needed).

⸻

6.  Logging & Monitoring
	•	Use Weights & Biases or local CSV for: loss curve, RAM usage (psutil), wall-clock.
	•	Save best checkpoint (lowest valid loss/ highest acc).
	•	Collect #trainable params via model.num_parameters(only_trainable=True).

⸻

7.  Evaluation & Analysis

7.1 Compute metrics:

ppl = math.exp(eval_loss)
acc = sklearn.metrics.accuracy_score(labels, preds)

7.2 Create plots:
	•	Accuracy / PPL vs effective model size (bytes).
	•	Rank allocation heat map (AdaLoRA trimmed ranks).
	•	Bar chart of peak RAM per run.

7.3 Stat tests: McNemar (SST-2) or paired t-test (ppl) to confirm joint variants ≠ baselines (α=0.05).

⸻

8.  Ablation / Diagnostics (optional, if time)
	•	Freeze AdaLoRA pruning ➜ fixed non-uniform rank, see if performance drops.
	•	Swap which layers get 8-bit in Joint-2.
	•	Try 3-bit (if hardware & bitsandbytes version allow) on the less-critical layers.

⸻

9.  Expected Outcomes
	•	Q4 alone ≈ FP16 (minor loss).
	•	AdaLoRA alone > fixed-rank FP16 (same params).
	•	Joint-2 hypothesized best efficiency-performance (keeps critical layers precise + adaptive rank).
	•	Memory footprint: Joint configs ≈ Q4 baseline; trainable % ≈ AdaLoRA (<0.15%).
	•	Training speed: Q4 runs slower/CPU-bound; FP16 fastest on MPS; note in results.

⸻

10.  Paper Artifacts
	•	Table 1: Metric comparison across configs.
	•	Figure 1: Pareto plot (bytes vs performance).
	•	Appendix A: Rank allocations produced by AdaLoRA.
	•	Appendix B: Mixed-precision map (layer → bit-width).
	•	Code + README: joint_adapt_rank_precision/.
