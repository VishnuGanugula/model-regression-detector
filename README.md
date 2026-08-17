# Model Regression Detector 🚨

An automated Machine Learning Operations (MLOps) pipeline designed to detect silent model regressions in Large Language Models (LLMs).

When updating prompts or fine-tuning weights, LLMs often suffer from catastrophic forgetting or regression—fixing one edge case while breaking another. This project provides a Continuous Integration (CI/CD) safety net by evaluating a custom PyTorch GPT-2 model against a deterministic Golden Dataset, calculating exact diffs, and blocking bad deployments via GitHub Actions.

## 🌟 Key Features

* **Custom Transformer Architecture:** A ground-up `CausalTransformer` built in PyTorch, mirroring the OpenAI GPT-2 (124M parameter) specification.
* **Supervised Fine-Tuning (SFT):** A training loop optimized to fine-tune the model on customer support classification datasets (JSONL) using Cross-Entropy Loss and the AdamW optimizer.
* **Deterministic Diff Engine:** Evaluates model outputs against frozen test cases to explicitly flag which specific outputs flipped from *Pass* to *Fail* between versions.
* **Automated CI/CD Pipeline:** Fully integrated GitHub Actions workflow that automatically spins up an Ubuntu runner, maps weights, executes the Diff Engine, and generates a visual HTML artifact on every push.

## 🏗️ Architecture & Stack

* **Core ML Framework:** PyTorch (`torch>=2.0.0`)
* **Model:** Custom Autoregressive Decoder-Only Transformer (GPT-2 124M Base)
* **CI/CD:** GitHub Actions (`.github/workflows/llm_eval.yml`)
* **Data Processing:** JSONL format for sequence ingestion
* **Reporting:** Auto-generated HTML/CSS Dashboards

## 📂 Repository Structure

```text
model-regression-detector/
├── .github/workflows/
│   └── llm_eval.yml          # Cloud CI/CD automation workflow
├── data/
│   ├── golden_dataset.json   # The frozen edge-cases for evaluation
│   ├── training_data.jsonl   # Kaggle synthetic dataset for SFT
│   └── run_history.json      # Historical logs of all evaluations
├── src/
│   ├── feature/
│   │   ├── evaluate.py       # The Diff Engine runner
│   │   ├── generate_report.py# HTML dashboard generator
│   │   ├── train.py          # Supervised Fine-Tuning loop
│   │   └── process_kaggle.py # CSV to JSONL data pipeline
│   ├── model/
│   │   ├── attention.py      # Multi-Head Attention blocks
│   │   ├── transformer.py    # Custom CausalTransformer class
│   │   └── tokenizer.py      # BPE Tokenizer wrapper
├── prompts/
│   └── v2.yaml               # Version-controlled system prompts
└── requirements.txt          # Python dependencies

```

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/VishnuGanugula/model-regression-detector.git
cd model-regression-detector
pip install -r requirements.txt

```

### 2. Loading Base Weights

Ensure the base GPT-2 124M weights (`custom_gpt2_124M.pth`) are placed in the root directory before running inference or fine-tuning.

### 3. Running the Diff Engine Locally

Evaluate the current model state against the Golden Dataset:

```bash
python src/feature/evaluate.py

```

Generate the visual regression dashboard:

```bash
python src/feature/generate_report.py

```

*Open `data/report.html` in any web browser to view the results.*

### 4. Fine-Tuning the Model

To adapt the model to a specific routing task using the JSONL dataset:

```bash
python src/feature/train.py

```

*This will output `fine_tuned_gpt2.pth` upon completion.*

## ☁️ Continuous Integration

This repository is equipped with an automated GitHub Actions pipeline. Upon pushing to the `main` branch or opening a Pull Request, the pipeline will:

1. Initialize an Ubuntu runner.
2. Install PyTorch and dependencies.
3. Execute the `evaluate.py` Diff Engine.
4. Generate the HTML report.
5. Upload `report.html` as a downloadable artifact directly to the GitHub interface.

---

**Author:** Ganugula Vishnu
