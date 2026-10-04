# CoNav code

This directory contains the CoNav implementation, organized into navigation
and 3D-language components.

## Structure

```text
code/
├── code_of_3D-LLM/
│   ├── model/
│   │   ├── point_navigator_3dllm.py
│   │   └── pointbert/
│   └── train/
└── code_of_nav_model/
    ├── models/
    ├── tasks/
    │   ├── agents/
    │   └── datasets/
    └── train.py
```

- `code_of_nav_model/` implements visual navigation models, task agents, and
  dataset interfaces.
- `code_of_3D-LLM/` contains the point-cloud language model, PointBERT
  components, and training utilities.

CoNav follows the method described in the [paper](https://arxiv.org/abs/2505.16663): a 3D-text model provides structured spatial-semantic hypotheses to a visual navigation agent, which integrates this information through cross-modal belief alignment.

## Environment setup

The component projects provide environment and installation instructions:

- [NaviLLM](https://github.com/zd11024/NaviLLM#installation) documents the
  navigation environment (Python 3.8.16), Matterport3D simulator setup, and
  dependencies.
- [PointLLM](https://github.com/InternRobotics/PointLLM) documents the 3D
  language-model environment (Python 3.10.13, PyTorch 2.0.1, CUDA 11.7, and
  Transformers development revision `cae78c46`).

The components use their respective upstream setups. Select a PyTorch and CUDA
build compatible with the target machine and follow the linked instructions
for simulator and package installation. ScanQA METEOR evaluation also uses a
JDK as described in the NaviLLM setup.

Example Conda environment for the navigation component:

```bash
conda create -n conav-nav python=3.8.16
conda activate conav-nav
python -m pip install --upgrade pip
```

## Data and model assets

Prepare Matterport3D and the benchmark datasets following the corresponding
dataset terms and the [NaviLLM data instructions](https://github.com/zd11024/NaviLLM).
Prepare point-cloud data and pretrained 3D-language model assets following the
[PointLLM instructions](https://github.com/InternRobotics/PointLLM). Configure
the data and checkpoint paths for the selected experiments before training or
evaluation.

## Related work

- [CoNav: Collaborative Cross-Modal Reasoning for Embodied Navigation](https://arxiv.org/abs/2505.16663)
- [Towards Learning a Generalist Model for Embodied Navigation](https://github.com/zd11024/NaviLLM)
- [PointLLM: Empowering Large Language Models to Understand Point Clouds](https://github.com/InternRobotics/PointLLM)

Please refer to each upstream repository for its license and acknowledgements.
