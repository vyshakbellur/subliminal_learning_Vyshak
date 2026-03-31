# Subliminal Learning (Fixed Prototype)

This prototype is the "full" approach you described: **DNA language modeling + a sample-specific latent code + adaptation**.
It is aligned with the "31-mers as words / sample as document" concept in your note.

## What it does

Given a set of FASTA/FASTA.GZ files where **each file is one sample**, the system produces:

- **Per-sample perplexity / NLL** (how surprised the model is by the sample).
- **Two sample embeddings**:
  - **Latent-code embedding (subliminal)**: an explicit per-sample vector learned (train) or inferred (eval).
  - **Pooled embedding (contextual)**: mean-pooled hidden states across the sample.
- **Adaptation diagnostics** for eval samples:
  - perplexity before adaptation
  - perplexity after adaptation
  - improvement (delta)

## Why this is "subliminal learning"

During training, the language model is optimized for next-token prediction *and* each training sample gets a learned
latent code that is injected as a prefix into the transformer. The model is never told environment labels.

For unseen samples, the **LM is frozen** and we optimize only the latent code to best explain the sample.
How much the loss improves (and how quickly) is a useful signal for domain shift vs. true out-of-domain novelty.

## Quickstart (CPU)

```bash
pip install -r requirements.txt
python scripts/run_demo.py
```

Outputs are written to:

```
outputs/demo_subliminal/
  config.json
  model.pt
  sample_ids.txt
  embeddings_latent.npy
  embeddings_pooled.npy
  samples_summary.csv
  samples_pca2.csv
```

## Running on your own samples

Train on in-domain samples, evaluate on everything:

```bash
python src/subliminal_sample_lm.py \
  --train-fasta data/samples/*.fna.gz \
  --eval-fasta  data/samples/*.fna.gz \
  --kmer 31 --stride 1 --vocab-size 32768 \
  --epochs 5 --d-model 64 --layers 2 --heads 4 \
  --train-block 256 --embed-block 512 --embed-step 512 \
  --latent-dim 64 --adapt-steps 60 --adapt-lr 0.2 \
  --save outputs/run1
```

## Notes

- This is a prototype intended for *validation of representation + novelty signals*, not a final predictive system.
- Embedding separation becomes meaningful once you have 10+ samples per environment and reasonable sequence depth.
