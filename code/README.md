# CoNav research code

This directory contains the recovered CoNav model and data pipeline files from
the local `source_code` snapshot. It is organized into two components:

- `code_of_nav_model/`: navigation models, task datasets, agents, and training
  entry point (`train.py`).
- `code_of_3D-LLM/`: point-cloud model and training components, including the
  PointBERT encoder configuration.

These files represent a partial recovery. The original repository's complete
configuration tree, launcher scripts, dataset preparation tools, dependency
lockfiles, checkpoints, and benchmark assets were not present in the recovered
snapshot. As a result, the code is provided for inspection and continued
development; end-to-end training and evaluation have not been verified from
this checkout. No checkpoints or datasets are included.

## Environment

The two upstream projects document different environments. NaviLLM documents
Python 3.8.16, while PointLLM's released setup documents Python 3.10.13,
PyTorch 2.0.1, CUDA 11.7, and a specific Transformers development revision.
Those environments have not been reconciled or validated together for this
recovered CoNav snapshot. Do not assume one environment can run both
components.

For the navigation component, begin with NaviLLM's environment and
installation guide, then install a PyTorch/CUDA build that matches the host
system. For the 3D-LLM component, use PointLLM's separately documented setup.
The Matterport3D simulator and benchmark data must also be installed and
configured for navigation. For ScanQA METEOR evaluation, NaviLLM documents an
additional JDK requirement.

```bash
conda create -n conav-nav python=3.8.16
conda activate conav-nav
python -m pip install --upgrade pip
```

Use one of these as a starting point, depending on the component you need:

- [NaviLLM installation and environment](https://github.com/zd11024/NaviLLM#installation)
- [PointLLM installation and environment](https://github.com/InternRobotics/PointLLM)

The exact dependency set for this CoNav snapshot cannot be reconstructed from
the files currently available. A unified, pinned `environment.yml` or
`requirements.txt` is therefore not supplied; installing guessed versions may
produce an unusable environment. The Python snippet above only creates a base
navigation environment; it does not install the simulator, PyTorch, or the
project's full dependencies.

## Code map

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

The high-level method is described in the [CoNav paper](https://arxiv.org/abs/2505.16663): a 3D-text model supplies structured spatial-semantic hypotheses to a
visual navigation agent, which learns to use these hypotheses through
cross-modal belief alignment. The recovered code reflects the project's
NaviLLM navigation and PointLLM-style 3D-language foundations; it should not be
treated as a complete reproduction package.

## Data and checkpoints

Datasets, Matterport3D assets, pretrained foundation models, and CoNav
checkpoints are not distributed here. Obtain each asset under its own license
and follow the corresponding benchmark and upstream project setup. Configure
paths in the restored experiment configuration before running training or
evaluation; those configuration files are not part of this snapshot.

## Upstream projects and attribution

- [CoNav](https://github.com/oceanhao/CoNav) — this project and its paper.
- [NaviLLM](https://github.com/zd11024/NaviLLM) — navigation-model foundation,
  based on *Towards Learning a Generalist Model for Embodied Navigation*.
- [PointLLM](https://github.com/InternRobotics/PointLLM) — 3D-language-model
  foundation, based on *PointLLM: Empowering Large Language Models to
  Understand Point Clouds*.

Consult the upstream repositories for their licenses and notices. The local
recovery did not include per-file provenance or license headers, so this
directory does not assign a new license to upstream-derived code.
