# Predicting Protein-Ligand Binding Affinity using Large Language Models 

This repository contains the code for predicting protein-ligand binding affinity using sequence-based embeddings generated by Large Language Models (LLMs), based on the BAPULM paper and subsequent explorations using ESM and ChemBERTa.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

## Overview

Predicting how strongly a potential drug (ligand) binds to its target protein is crucial in drug discovery. This project implements and explores a deep learning framework that predicts binding affinity ($pK_d$) using only protein amino acid sequences and ligand SMILES strings.

The core idea is to leverage powerful pre-trained LLMs (like ProtT5, MolFormer, ESM, ChemBERTa) as feature extractors to convert sequences into numerical embeddings. A downstream feed-forward neural network then uses these embeddings to predict the binding affinity. This approach avoids the need for complex 3D structural data, offering a faster and more broadly applicable method for virtual screening.

## Motivation

* **Challenge:** Traditional experimental and 3D computational methods for affinity prediction can be slow, costly, or require data (3D structures) that isn't always available.
* **Opportunity:** Utilize the power of LLMs, pre-trained on vast biological/chemical data, to extract meaningful features directly from sequences for efficient and scalable affinity prediction.

## Architecture

The pipeline consists of two main stages:

1.  **LLM Feature Extraction:**
    * Protein sequences are fed into a protein LLM (e.g., ESM, ProtT5).
    * Ligand SMILES strings are fed into a ligand LLM (e.g., ChemBERTa, MolFormer).
    * These LLMs (used as frozen feature extractors) output fixed-size embedding vectors.
    * *(For training efficiency, these embeddings are pre-computed and stored in a JSON file.)*
2.  **Predictive Neural Network:**
    * The protein and ligand embeddings are projected to a common dimension (e.g., 512).
    * The projected embeddings are concatenated.
    * A feed-forward neural network (MLP with BatchNorm, Dropout, ReLU) processes the combined vector to predict a single scalar value ($pK_d$).

<img width="353" alt="Screenshot 2025-05-12 at 20 48 09" src="https://github.com/user-attachments/assets/730a92c4-275a-4707-8572-1002d6f80bbb" />


## Features

* Sequence-based: Relies only on protein sequences and SMILES strings.
* LLM-Powered: Leverages state-of-the-art pre-trained language models.
* Efficient Training: Uses pre-computed embeddings to speed up the training of the predictive network.
* Modular Code: Organized structure for data loading, model definition, training, and inference.

## Setup Instructions

1.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have PyTorch installed with CUDA support if using a GPU.)*

2.  **Download Pre-computed Embeddings:**
    * You need the JSON file containing the pre-computed embeddings corresponding to the LLMs your model expects.
    * **Example for ESM + ChemBERTa:** Run the dataset_creation.ipynb to create the binding_affinity_100k_raw.csv. Then run the `dataset_creation.py` to create `esm_chemberta_embeddings.json`
    * **Example for Original BAPULM:** Download `prottrans_molformer_embeddings.json` from https://huggingface.co/datasets/radh25sh/BAPULM/tree/main
    * **Place the downloaded `.json` file inside the `data/` directory.** The filename should match the `dataset_path` specified in `config.yaml`.

## Usage

**Configuration:**

* Modify `config.yaml` to set hyperparameters (learning rate, batch size, epochs), file paths (`dataset_path`, `model_train_save_path`), device (`cuda` or `cpu`), etc.
* **Important:** Ensure the `dataset_path` in `config.yaml` points to the correct pre-computed embeddings JSON file you downloaded/created.

**Training:**

* Make sure your `model/model.py` file defines the `Model` class with input dimensions matching the embeddings in your JSON file (e.g., 1280 for esm2_t33_650M_UR50D, 640 for esm2_t30_150M_UR50D, 1024 for ProtT5, 768 for Molformer and seyonec/ChemBERTa-zinc-base-v1, 384 for DeepChem/ChemBERTa-77M-MLM...).
* Run the training script:
    ```bash
    python main.py
    ```
* Trained model weights will be saved to the path specified by `model_train_save_path` in `config.yaml`

**Inference:**

* Ensure `utils/preprocessing.py` contains the `EmbeddingExtractor` configured with the *same* LLMs used to train the model weights you are loading (e.g., ESM and ChemBERTa).
* Make sure the `model_train_save_path` in `config.yaml` points to the trained model weights you want to use for inference.
* Ensure the benchmark files listed in `benchmark_files` within `config.yaml` exist in the `data/` directory.
* Run the inference script:
    ```bash
    python inference.py
    ```
* The script will load the specified benchmark files, generate embeddings on-the-fly using the `EmbeddingExtractor`, predict affinities using the loaded model, and print results (potentially including metrics if true values are available in the benchmark files).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

* Authors of the original BAPULM paper.
* Creators of the underlying LLMs (ProtT5, MolFormer, ESM, ChemBERTa).
* Maintainers of the Hugging Face Transformers and Datasets libraries.
* Providers of the benchmark datasets.

