# Criminal Inductive Link Prediction
Inductive Link Prediction for Criminal Network Analysis

This work has been adapted from the original implementation by `working-yuhao`, found at https://github.com/working-yuhao/DEAL, and a fork by `lajd`, found at https://github.com/lajd/DEAL.

Specifically, this project involves restructuring the initial implementation to make it more concise and modular. The restructuring process is solely dedicated to enhancing the inductive link prediction task for criminal network.

### Requirements
- Linux
- Nvidia GPU
- Cuda version 11.0
- Anaconda

### Installation
The installation below uses anaconda.

```shell
#!/bin/bash

CONDA_ENV_NAME=DEAL

#conda clean --all -y
conda env list | grep ${CONDA_ENV_NAME}
if [ $? -eq 0 ]; then
    echo "DEAL environment already exists; skipping creation"
else
    echo "Creating ${CONDA_ENV_NAME} environment"
    conda create -n ${CONDA_ENV_NAME} python=3.9 -y
fi

conda activate ${CONDA_ENV_NAME}

conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -c nvidia -y
conda install -q -y numpy pyyaml scipy ipython mkl mkl-include conda-build
conda install pyg -c pyg -c conda-forge -y

pip install -e .
```


### Reproduction

#### Burglary Dataset

```shell
make train-burglary
```

```yaml
 Total Load data time: 58.32 s
 Total Train/val time: 78.44 s
 Test time: 0.01 s
 Total time: 136.76 s
 ROC-AUC: 0.7580 
 AP: 0.7567
```

```shell
make train-burglary-ind-val
```

```yaml
 Total Load data time: 36.09 s
 Total Train/val time: 66.05 s
 Test time: 0.01 s
 Total time: 102.15 s
 ROC-AUC: 0.7468 
 AP: 0.7477
```

### Re-generation of JSON graph file from raw burglary dataset (Optional)
```python
python israel_lea_inp_burglary_v2_crimes_network.py
```
