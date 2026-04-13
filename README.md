# 2026Q2_innovation_sprint

# Installation
UV is used to manage the project's dependencies. Installation of the project requires Python >=3.13.

```
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

In order to run the notebook `segmentation.ipynb`, a local copy of the DinoV3 repository needs to be added to the root of this repository: https://github.com/facebookresearch/dinov3.git

Some notebooks require a HuggingFace token, this can be placed in a `.env` file following the format of `.sample_env`.