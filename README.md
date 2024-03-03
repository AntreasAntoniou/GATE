# Generalization After Transfer Evaluation (GATE)

GATE is a comprehensive benchmarking suite that aims to fill a gap in the evaluation of foundation models. Typically, foundation model evaluation overlooks the diverse applicability found in real-world settings. GATE is designed for multi-modal and multi-task scenarios, stress-testing models in a variety of domains beyond their initial training.

## Features

- Facilitates the transfer of neural network trunks across different modalities, domains, and tasks, promoting robust fine-tuning for visual tasks.
- Maximizes research signal per GPU hour through carefully selected scenarios.
- Enables straightforward replacement of transferable trunks with minimal effort.

## Installation

You can install GATE using pip:

```bash
pip install gate
```

To install from source, clone the repository and use the provided requirements file:

```bash
git clone https://github.com/yourusername/gate.git
cd gate
pip install -r requirements.txt
pip install -e . # for local dev editable mode
or 
pip install .
```

For development purposes, use the requirements_dev.txt file:

```bash
pip install -r requirements_dev.txt
```

## Usage

#### Use GATE as a template for your research project

GATE can be used as a template for your research project. It provides full Hydra integration and includes boilerplate, models, datasets, trackers, and more. Note that using GATE as a template may involve a lot of overhead and time to learn, and may be complex.

#### Use GATE as a library

GATE can be used as a library in your Python projects. Here is a basic example:

```python
import gate.data.image.classification.stl10 as stl
data = stl.build_stl10_dataset("train", data_dir=os.environ.get("PYTEST_DIR"))

import gate.models.classification.clip as clip
model = clip.build_clip_model("RN50x4", pretrained=True)
```

#### Use GATE as a library, as a source of experiment generation

GATE can be used as a library to generate experiments. Here is an example:

```python
builder = gate.build_experiments(model=GATEModel(), gate_flavour="foundation")
experiments = builder.generate_experiments()
builder.run_experiments()
```

## Project Structure

A high-level overview of the project's structure is given below:

```
.
├── boilerplate/
│   ├── callbacks.py
│   ├── convenience.py
│   ├── core.py
│   ├── decorators.py
│   ├── utils.py
│   └── wandb_utils.py
├── config/
│   ├── config.py
│   └── variables.py
├── data/
│   ├── few_shot/
│   ├── image/
│   ├── image_text/
│   ├── medical/
│   ├── tasks/
│   ├── transforms/
│   ├── video/
│   └── core.py
├── menu/
│   ├── configs/
│   ├── builder.py
│   ├── collector.py
│   ├── core.py
│   └── utils.py
├── metrics/
│   ├── core.py
│   ├── glossary.py
│   ├── multi_class_classification.py
│   ├── segmentation.py
│   └── vqa_eval.py
├── models/
│   ├── backbones/
│   ├── blocks/
│   ├── task_adapters/
│   └── core.py
├── orchestration/
│   ├── evaluators/
│   ├── trainers/
│   └── utils/
├── dummy_module.py
└── run.py
```

For a more detailed description of the individual files and directories, please refer to the comments in the respective files.
