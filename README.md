# Inductive Link Prediction for Criminal Network Analysis
[![python](https://img.shields.io/badge/python-3.9.16-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8.1-orange)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.0.1-green)](https://pytorch-geometric.readthedocs.io/)
[![NetworkX](https://img.shields.io/badge/networkx-2.6.3-orange)](https://networkx.org/)
[![ROXANNE-license](https://img.shields.io/badge/License-ROXANNE-blue.svg)](https://www.roxanne-euproject.org/)

This repository is a part of the paper "Inductive and Transductive Link Prediction for Criminal Network Analysis," published in the Journal of Computational Science in 2023. The paper discusses how identifying potential offenders who might co-offend can help law enforcement focus their investigations and improve predictive policing. Traditional methods rely heavily on manual work by police officers, which can be inefficient. To address this, the paper introduces two machine learning frameworks based on graph theory, specifically for burglary cases. These are transductive link prediction (see the implementation [here](https://github.com/erichoang/criminal-network-visualization)), which predicts connections between existing nodes (offenders or cases), and inductive link prediction (this repository), which finds links between new cases and existing nodes.

This repository has been adapted from the original implementation by `working-yuhao`, found at https://github.com/working-yuhao/DEAL, and a fork by `lajd`, found at https://github.com/lajd/DEAL. Specifically, this repository involves restructuring the initial implementation to make it more concise and modular. The restructuring process is dedicated to enhancing criminal networks' inductive link prediction task.

## Citation
 Ahmadi, Z., Nguyen, H. H., Zhang, Z., Bozhkov, D., Kudenko, D., Jofre, M., Calderoni, F., Cohen, N., & Solewicz, Y. (2023). Inductive and transductive link prediction for criminal network analysis. Journal of Computational Science. [Preprint](https://hoanghnguyen.com/assets/pdf/ahmadi2023inductive.pdf)
```
@article{ahmadi2023inductive,
  title = {Inductive and Transductive Link Prediction for Criminal Network Analysis},
  author = {Ahmadi, Zahra and Nguyen, Hoang H. and Zhang, Zijian and Bozhkov, Dmytro and Kudenko, Daniel and Jofre, Maria and Calderoni, Francesco and Cohen, Noa and Solewicz, Yosef},
  journal = {Journal of Computational Science},
  publisher = {Elsevier},
  volume = {72},
  pages = {102063},
  year = {2023},
  issn = {1877-7503},
  doi = {https://doi.org/10.1016/j.jocs.2023.102063},
  url = {https://www.sciencedirect.com/science/article/pii/S1877750323001230},
}
```
 

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
