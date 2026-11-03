# LSC-Parlament: Large-Scale Catalan Sign Language Translation

[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-blue)](https://huggingface.co/datasets/LangTech-MT/LSC-Parlament)


Official implementation of the paper: **"LSC-Parlament: An Automatically Aligned Catalan Sign Language Dataset from Parliament Videos."**

---

##  Overview
We introduce **LSC-Parlament**, a large-scale, multimodal dataset for Catalan Sign Language built from parliamentary sessions recorded between 2021 and 2024. The dataset is constructed through a fully automated pipeline that leverages audio timestamps as an intermediate link to align sign language video with spoken-language transcriptions, eliminating the need for any prior human annotation. While demonstrated on parliamentary data, the pipeline is designed to be applicable to any broadcast source containing interpreted sign language. The resulting corpus contains over 87 hours of video across two spoken-language tracks (Catalan or Spanish) each densely aligned at the segment level with the corresponding interpreter signing, speech recording, and written translation. Unlike most existing sign language corpora, which are limited to video-text pairs, LSC-Parlament jointly offers RGB video, audio, and text, covering open-domain parliamentary discourse and enabling research in both multimodal and cross-lingual SLT.

##  Installation

```bash
# Clone the repository
git clone [https://github.com/username/LSC-Parlament.git](https://github.com/username/LSC-Parlament.git)
cd LSC-Parlament

# Create environment
conda create -n lsc-parlament python=3.10
conda activate lsc-parlament

# Install dependencies (Linux/SLURM setup)
sh install_dependencies.sh
```

### Tested local setup (macOS arm64, CPU only)

`install_dependencies.sh` targets Linux (`apt-get`) and references `requirements*.txt` files that are not in the repository. The following setup was tested end to end on a single session. The pins matter: the code uses NeMo's 1.x `transcribe(paths2audio_files=..., logprobs=True)` API and ffmpeg's `-vsync` option (removed in ffmpeg 7).

```bash
conda create -n lsc-parlament -c conda-forge python=3.10 "ffmpeg=6"
conda activate lsc-parlament

# NeMo 1.22 (its repo is also needed for the forced aligner tool)
git clone --depth 1 --branch v1.22.0 https://github.com/NVIDIA/NeMo.git
pip install "numpy<1.24" Cython "torch==2.1.2" "torchaudio==2.1.2"
cat NeMo/requirements/requirements{,_asr,_common,_lightning}.txt | sed 's/#.*//' | grep -v -e triton -e youtokentome -e '^\s*$' > nemo_reqs.txt
pip install -r nemo_reqs.txt
pip install --no-build-isolation youtokentome
pip install --no-deps "nemo_toolkit==1.22.0"   # triton has no macOS wheels; skip it

# Pipeline dependencies
pip install fasttext-wheel pyctcdecode https://github.com/kpu/kenlm/archive/master.zip jiwer pandas \
    minio s3path python-dotenv inflect scraperwiki lxml pyyaml fuzzywuzzy python-Levenshtein cdifflib \
    pymongo opencv-python pympi-ling video2numpy pose-format "mediapipe==0.10.14"
# Compatibility pins for NeMo 1.22 / mediapipe
pip install "numpy<1.24" "setuptools<70" "huggingface_hub==0.23.5" "transformers==4.40.2" \
    "datasets<2.20" "onnx<1.17" "tensorboard<2.16"
```

`scraperwiki` needs `pdftohtml` (poppler) on the `PATH`.

### External assets

| Asset | Location expected by the code | How to get it |
|---|---|---|
| Sign segmentation model (Moryossef et al.) | `sign_language_segmentation/` (repo root) | `git clone https://github.com/sign-language-processing/segmentation.git sign_language_segmentation_repo && git -C sign_language_segmentation_repo checkout 4b3a0b6~1 && ln -s sign_language_segmentation_repo/sign_language_segmentation sign_language_segmentation`. Later versions removed `dist/model_E1s-1.pth` and `src/utils/probs_to_segments.py`. |
| FastText language ID | `data/models/lid.176.ftz` | `curl -L -o data/models/lid.176.ftz https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz` |
| NeMo ASR models | `nemo_models/stt_{ca,es}_conformer_ctc_large.nemo` | `nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_ca_conformer_ctc_large").save_to(...)`, same for `es` |
| KenLM language model | `data/models/Parla_Text_Corpus/5GRAM_Parla_Text_Corpus.bin` | Not publicly released. Any KenLM model (binary or ARPA, order ≥ 2) works. It only affects the auxiliary `cer_score_lm`; the CER filter uses `cer_score`. |

## Usage

The code is structured into the following steps:

0.  **Visual filtering:** Flag videos without sign language interpretation by comparing a 16×16 patch of the interpreter overlay against reference videos (3σ outlier frames, >10% → flagged). Flagged videos are reported in a JSON file for manual verification, not deleted.
1.  **Convert data:** Format all metadata in **YAML** format for use in subsequent pipeline stages.
2.  **Audio extraction:** Extract audio in `.wav` format from the downloaded videos.
3.  **Audio alignment:** Match diary speaker turns to the session metadata (fuzzy matching on speaker names) to get one timestamped block per speaker turn.
4.  **Audio segmentation:** Cut the audio into those blocks.
5.  **Video segmentation:** Cut the matching video blocks, cropped to the 330×330 interpreter overlay.
6.  **Dataset generation:** Per block, force-align the diary text to the audio with the NeMo Forced Aligner (sentence-level segments), run language ID and ASR on each segment to compute its CER, segment the signing video with the **pretrained pose-based segmenter**, and pair audio and video segments (Algorithm 1, 6 s fallback).
7.  **Casing restoration:** Recover the cased, punctuated diary text for each segment (`src/restore_casing_punctuation.py`).
8.  **Filtering and deduplication:** Keep segments with CER ≤ 0.3, remove exact-text duplicates, and write the train/validation/test TSVs (`src/segments_to_train_dataset.py`).

### Running locally

Steps 2–6 go through `src/clusters_launcher.py`. The `N-run-*.sh` scripts submit them with `--sbatch`; to run without SLURM, drop `--sbatch` and pass a session id. Run from the repository root with `PYTHONPATH` set so that `sign_language_segmentation` can be imported. Example for a session `<SID>` stored in `data/<SID>/{videos,jsons,pdfs}`:

```bash
export PYTHONPATH=$PWD
D=data/<SID>; S=<SID>
python src/visual_filtering.py -v $D/videos -r <validated reference videos>.mp4 -o $D/visual_filtering.json   # 0
python src/convert_data.py --source pdf --dest yaml --in_dir $D/pdfs --out_dir $D/yamls --process_xml          # 1
for p in audio_extraction alignment segmentation video_segmentation; do                                        # 2-5
    python src/clusters_launcher.py --process $p --data-directory $D --session-id $S
done
python src/clusters_launcher.py --process dataset_generation --data-directory $D --session-id $S \
    --ca-nemo-model nemo_models/stt_ca_conformer_ctc_large.nemo \
    --es-nemo-model nemo_models/stt_es_conformer_ctc_large.nemo                                                # 6
(cd $D && python ../../src/restore_casing_punctuation.py)                                                      # 7
(cd $D && python ../../src/segments_to_train_dataset.py -i final_output_restored -s video_segments \
    -o dataset -c 0.3 -l ca es)                                                                                # 8
```

Step 6 dominates the runtime: about 1.3–1.5× real time on CPU. It also writes the raw frames of every video block to `video_segments/<SID>/frames/` (about 13 GB per hour of video), and the pipeline never reads them; they can be deleted after each session.

---

Once the dataset is created, models can be trained using the **MultimodalHugs** framework:

9.  **Metadata creation:** Convert the generated dataset into the metadata format required by the framework.
10. **Training:** The default model configuration utilizes **CLIP** as the feature extractor and **M2M-100 418M** as the translation model.
11. **Evaluation:** Assess the model's performance on the designated **test partition**.

These correspond to the `7-`, `8-` and `9-run-*.sh` scripts. For full commands, argument descriptions, and SLURM wrappers, see [docs/slt_experiments.md](docs/slt_experiments.md).


