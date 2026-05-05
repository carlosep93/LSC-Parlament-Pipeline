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

# Install dependencies
sh install-dependencies.sh
```

## Usage

The code is structured into the following steps:

1.  **Convert data:** Format all metadata in **YAML** format for use in subsequent pipeline stages.
2.  **Audio extraction:** Extract audio in `.wav` format from the downloaded videos.
3.  **Audio alignment:** Synchronize audio fragments with their corresponding transcriptions.
4.  **Audio segmentation:** Partition audio files to match transcribed segments.
5.  **Video segmentation:** Segment video files using a **pretrained segmenter**.
6.  **Dataset generation:** Align the audio and video segments to create the final dataset.

---

Once the dataset is created, models can be trained using the **MultimodalHugs** framework:

7.  **Metadata creation:** Convert the generated dataset into the metadata format required by the framework.
8.  **Training:** The default model configuration utilizes **CLIP** as the feature extractor and **M2M-100 418M** as the translation model.
9.  **Evaluation:** Assess the model's performance on the designated **test partition**.

For full commands, argument descriptions, and SLURM wrappers for steps 7–9, see [docs/slt_experiments.md](docs/slt_experiments.md).


