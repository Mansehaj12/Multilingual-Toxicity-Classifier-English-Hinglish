# Project Overview: Multilingual Toxicity Classification

This document provides a comprehensive explanation of what this project is and details the function and contents of every file in the repository.

---

## What is this Project?

Social media platforms face significant challenges in moderating toxic content (such as hate speech, offensive language, and threats) at scale. This problem is particularly acute in India, where users frequently communicate using **Hinglish** (a code-mixed combination of Hindi and English written in the Roman/Latin script, e.g., *"tu bahut stupid hai"*). Standard monolingual English classifiers treat Hinglish grammar and transliterated Hindi words as noise, leading to poor moderation performance.

This project is a **comparative deep learning study** designed to address this problem. It evaluates three different generations of Transformer architectures on a unified corpus of **354,895 examples** containing English and Hinglish posts:

1. **mBERT (`bert-base-multilingual-cased`)**: An encoder-only multilingual model from 2019 (178M parameters, fully fine-tuned).
2. **XLM-RoBERTa (`xlm-roberta-base`)**: A stronger encoder-only model from 2020 pre-trained on a much larger multilingual corpus (278M parameters, fully fine-tuned).
3. **Qwen2.5-0.5B + LoRA**: A modern 2024-era decoder-only Large Language Model (494M parameters, with 1.1M parameters trained using Low-Rank Adaptation).

The models are evaluated on **six toxic labels**: `toxic`, `obscene`, `insult`, `identity_hate`, `threat`, and `severe_toxic`. The project also includes an **explainability layer** using **gradient-based token attribution** to highlight which specific words (toxic spans) triggered the toxicity prediction.

---

## File-by-File Breakdown

Here is what is happening inside each file in this repository:

### 1. [bert-base-multilingual.ipynb](file:///c:/Users/HP/OneDrive/Desktop/Codes/Projects%20All/Toxicity-classification/bert-base-multilingual.ipynb)
This Jupyter Notebook contains the pipeline for training and evaluating the **mBERT** model:
- **Data Harmonization & Merging**: Loads and cleans three datasets: Jigsaw Toxic Comment Classification (English), HASOC 2019–2021 (English/Hindi), and MMHS150K (English tweets). It maps their labels to the unified 6-label schema and merges them into a shuffled dataset.
- **Model Training**: Sets up a multi-label classification head on top of mBERT and runs training using the Hugging Face `Trainer` API for 5 epochs.
- **Evaluation**: Computes F1 Micro, F1 Macro, F1 Weighted, Precision, Recall, and ROC-AUC metrics on a held-out validation set.
- **Visualizations**: Saves plots for loss curves, confusion matrices per label, ROC curves, and per-label F1 scores.

### 2. [xlm-r-merged.ipynb](file:///c:/Users/HP/OneDrive/Desktop/Codes/Projects%20All/Toxicity-classification/xlm-r-merged.ipynb)
This Jupyter Notebook contains the training, evaluation, and explainability pipeline for the **XLM-RoBERTa** model:
- **Data Prep & Model Fine-tuning**: Preprocesses the data, initializes `xlm-roberta-base` for multi-label classification, and trains it for 5 epochs.
- **Evaluation**: Evaluates the model on the validation split and exports its final metrics. It achieves the best F1 Macro (0.6525) among the models, indicating it handles rare classes (`threat`, `severe_toxic`) much better than the LoRA model.
- **Gradient-Based Attribution (Newly Added)**: Computes the gradient of the predicted `toxic` label logit with respect to the input word embeddings. It sums the absolute gradients across the hidden dimension, normalizes the scores to `[0, 1]`, and maps them back to the input words using token offsets. This allows us to highlight exactly which words (e.g., *"bahut"*, *"stupid"*) contributed most to a "toxic" classification.

### 3. [qwen2-5-merged.ipynb](file:///c:/Users/HP/OneDrive/Desktop/Codes/Projects%20All/Toxicity-classification/qwen2-5-merged.ipynb)
This Jupyter Notebook trains and evaluates the decoder-only **Qwen2.5-0.5B** model:
- **Parameter-Efficient Tuning (LoRA)**: Instead of updating all 494 million weights, it freezes the base LLM and attaches low-rank adapters (rank $r=8$) to the query and value projection layers (`q_proj`, `v_proj`). Only 1.1 million parameters (0.22% of the model) are trained.
- **Evaluation**: Computes the same metrics. Qwen2.5 + LoRA achieves the highest F1 Micro (0.7362) because of higher recall on common labels, but its performance falls off on extremely rare labels due to the small capacity of the adapter.

### 4. [cross-model-comparison.ipynb](file:///c:/Users/HP/OneDrive/Desktop/Codes/Projects%20All/Toxicity-classification/cross-model-comparison.ipynb)
This Jupyter Notebook aggregates the metrics from the three models and generates comprehensive comparative visualizations:
- **Overall F1 Comparison Bar Chart**: Compares Micro, Macro, and Weighted F1 scores.
- **Per-Label F1 Bar Chart**: Shows how each model fares on each of the six labels.
- **Model Heatmap**: Provides a clean visual grid of scores.
- **ROC-AUC Comparison**: Visualizes ranking performance (where all models score $>0.97$ on rare labels, showing the deficiency is a threshold calibration issue rather than a model capability issue).
- **Radar Chart**: Maps the relative strengths and weaknesses of each model in a polar projection.

### 5. [Toxicity_Classification_Report.pdf](file:///c:/Users/HP/OneDrive/Desktop/Codes/Projects%20All/Toxicity-classification/Toxicity_Classification_Report.pdf)
A professional 33-page academic report covering:
- Formal mathematical formulations and derivatives.
- Comprehensive literature review and dataset background.
- Thorough analysis of training dynamics, confusion matrices, and model tradeoffs.
- Extensive error analysis (including challenges like sarcasm, identity mentions, and veiled threats).
- Full details of the gradient attribution module and references.

### 6. [README.md](file:///c:/Users/HP/OneDrive/Desktop/Codes/Projects%20All/Toxicity-classification/README.md)
The repository home page that introduces the project, details performance tables, lists limitations, explains how to reproduce the results, and cites the corresponding paper.
