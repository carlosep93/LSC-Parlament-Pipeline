# SLT Baseline Experiments

This document covers steps 7–9 of the pipeline: converting the aligned dataset into
the format expected by [MultimodalHugs](https://github.com/GerrySant/multimodalhugs),
training the SLT baseline models, and evaluating them.

All commands below are run from the repository root. For cluster environments, each
step has a corresponding SLURM wrapper in `scripts/` that accepts the same arguments.
Before running, set `--account` and `--qos` in those wrappers to match your cluster
allocation.

---

## Prerequisites

Install MultimodalHugs following its own installation instructions. The experiments
in the paper used **MultimodalHugs v0.4.2**.

---

## Step 7 — Metadata Creation

Converts the JSON alignment output (produced by step 6) into TSV metadata files
required by MultimodalHugs.

```bash
python src/segments_to_train_dataset.py \
    --in_dir        final_output/ \
    --in_dir_video  video_segments/ \
    --out_dir       metadata/ \
    --max_cer_score 0.3 \
    --languages     <lang> \
    --cased \
    --filter_sessions
```

Run once per language track (`<lang>` ∈ `{ca, es}`). For a bilingual experiment,
pass both codes at once: `--languages ca es`.

**Key arguments:**

| Argument | Description |
|---|---|
| `--max_cer_score 0.3` | Discard segments whose Character Error Rate against the ASR reference exceeds 0.3 |
| `--languages` | One or more spoken-language codes to include (`ca`, `es`) |
| `--cased` | Preserve original casing (do not lowercase transcriptions) |
| `--filter_sessions` | Exclude a fixed set of sessions with known recording or annotation artefacts |

The retained segments are randomly split into **train / dev / test**
(60 % / up to 2,000 / up to 2,000 samples, seed 42).

Each output TSV has the following columns:

| Column | Description |
|---|---|
| `signal` | Path to the video file |
| `signal_start` | Segment start in milliseconds |
| `signal_end` | Segment end in milliseconds |
| `encoder_prompt` | Sign language tag (e.g. `__lsc__`) |
| `decoder_prompt` | Target language tag (e.g. `__ca__`, `__es__`) |
| `output` | Reference transcription |

**SLURM wrapper:**
```bash
sbatch scripts/run_metadata_creation_multimodalhugs.sh \
    --in_dir final_output/ \
    --in_dir_video video_segments/ \
    --out_dir metadata/ \
    --max_cer_score 0.3 \
    --languages <lang> \
    --cased \
    --filter_sessions
```

---

## Step 8 — Training

Initialises the processor, model, and dataset artefacts, then fine-tunes end-to-end.
Each language experiment has its own config in `configs/` (e.g. `configs/config_ca.yaml`,
`configs/config_es.yaml`). The configs differ only in a small set of
language-specific fields; all training hyperparameters are shared.

```bash
# Initialise processor, model, and dataset artefacts
multimodalhugs-setup --modality video2text \
    --config_path configs/config_<lang>.yaml

# Fine-tune
multimodalhugs-train --task seq2seq \
    --config_path            configs/config_<lang>.yaml \
    --model_name_or_path     <setup_dir>/model \
    --processor_name_or_path <setup_dir>/video2text_processor \
    --dataset_dir            <setup_dir>/datasets/parlament_<lang>_video \
    --output_dir             trained/<run_id>
```

Where `<setup_dir>` is the directory created by `multimodalhugs-setup`
(controlled by `setup.output_dir` in the YAML config).

**Architecture:** CLIP ViT-B/32 visual encoder → linear projection with
pre-LayerNorm → M2M-100 418M seq2seq backbone. Uniform temporal downsampling
keeps every second frame. All parameters are trained end-to-end.

**SLURM wrapper:**
```bash
sbatch scripts/run_training_multimodalhugs.sh \
    configs/config_<lang>.yaml \
    metadata/ \
    <lang>
```

---

## Step 9 — Evaluation

Evaluation runs in two stages.

### Stage 1 — SacreBLEU and ChrF

```bash
multimodalhugs-generate --task translation \
    --metric_name            sacrebleu \
    --generate_output_dir    results/sacrebleu \
    --model_name_or_path     <checkpoint> \
    --processor_name_or_path <setup_dir>/video2text_processor \
    --dataset_dir            <setup_dir>/datasets/parlament_<lang>_video \
    --config_path            configs/config_<lang>.yaml

multimodalhugs-generate --task translation \
    --metric_name            chrf \
    --generate_output_dir    results/chrf \
    --model_name_or_path     <checkpoint> \
    --processor_name_or_path <setup_dir>/video2text_processor \
    --dataset_dir            <setup_dir>/datasets/parlament_<lang>_video \
    --config_path            configs/config_<lang>.yaml
```

**SLURM wrapper:**
```bash
sbatch scripts/run_evaluate_multimodal_hugs.sh \
    configs/config_<lang>.yaml \
    metadata/ <lang> results/ <checkpoint> sacrebleu

sbatch scripts/run_evaluate_multimodal_hugs.sh \
    configs/config_<lang>.yaml \
    metadata/ <lang> results/ <checkpoint> chrf
```

### Stage 2 — Neural metrics

BERTScore, COMET-Kiwi, LaBSE, and BLASER are computed from the predictions file
written by Stage 1 (`results/sacrebleu/predictions_labels.txt`):

```bash
python src/evaluate_metrics.py \
    --in_file  results/sacrebleu/predictions_labels.txt \
    --out_path results/ \
    --metrics  bertscore comet labse blaser \
    --source_language <lang> \
    --target_language <lang>
```

Model checkpoints used:

| Metric | Model |
|---|---|
| BERTScore | `microsoft/deberta-xlarge-mnli` |
| COMET-Kiwi | `Unbabel/wmt23-cometkiwi-da-xl` |
| LaBSE | `sentence-transformers/LaBSE` |
| BLASER 2.0 QE | `blaser_2_0_qe` |

> **Note:** `Unbabel/wmt23-cometkiwi-da-xl` requires accepting the model terms on
> HuggingFace and authenticating with `huggingface-cli login` before use.

**SLURM wrapper:**
```bash
sbatch scripts/run_evaluate_other_metrics.sh \
    --in_file results/sacrebleu/predictions_labels.txt \
    --out_path results/ \
    --metrics bertscore comet labse blaser \
    --source_language <lang> \
    --target_language <lang>
```
