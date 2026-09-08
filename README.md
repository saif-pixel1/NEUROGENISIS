# HELIX — Self-Evolving Machine Learning

**HELIX (Hierarchical Evolutionary Learning & Intelligent eXaptation)** is a prototype framework for machine-learning systems that adapt to changing data streams while preserving previously learned knowledge.

Inspired by biological evolution and DNA, HELIX combines **concept drift detection, adaptive learning strategies, and replay memory** to continuously update a model as new environments arrive.

## Dataset
https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/index.html#expression_matrices/Developing-Mouse-Vis-Cortex-10X/20260131/
Public URl:
https://alleninstitute.github.io/abc_atlas_access/descriptions/Dev-Mouse-Vis-Cortex-10X.html

The project uses the **Developing Mouse – Visual Cortex (Dev-Mouse-Vis-Cortex-10X)** dataset from the **Allen Brain Cell Atlas**.

The dataset contains single-cell gene-expression data across multiple developmental stages, allowing each stage to be treated as a sequential streaming environment.

* Expression matrices
* Cell and gene metadata
* Developmental age information
* Cell-type annotations

## Project Structure

```text
helix_self_evolving/
├── data/
│   └── Allen dataset files
├── src/
│   ├── data_loader.py
│   ├── drift_detection.py
│   ├── memory.py
│   ├── models.py
│   ├── experiment.py
│   └── utils.py
├── notebooks/
├── requirements.txt
└── run_experiment.py
```

## Core Components

**Data Loader** — Loads and preprocesses the Allen dataset.

**Drift Detection** — Detects changes in the incoming data stream using statistical methods.

**Replay Memory** — Stores important previous samples to reduce forgetting.

**Model Wrappers** — Implements static, naive streaming, and HELIX models.

**Experiment Runner** — Runs the streaming experiment and compares model performance.

## Pipeline

```text
Allen Dataset
      ↓
Preprocessing
      ↓
Streaming Data
      ↓
Drift Detection
      ↓
Strategy Selection
      ↓
Model Adaptation
      ↓
Replay Memory
      ↓
Evaluation
```

## Installation

```bash
git clone <repository-url>
cd helix_self_evolving
pip install -r requirements.txt
```

## Run

```bash
python run_experiment.py
```

The experiment produces performance results for the **Static**, **Naive Streaming**, and **HELIX** approaches.

## Status

HELIX is an experimental research prototype for studying adaptive machine learning under changing data distributions.

## License

Add your preferred license here.
