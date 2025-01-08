<!-- # TGB -->
![TGB logo](imgs/logo.png)

**Temporal Graph Benchmark for Machine Learning on Temporal Graphs** (NeurIPS 2023 Datasets and Benchmarks Track)
<h3>
  <a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/066b98e63313162f6562b35962671288-Abstract-Datasets_and_Benchmarks.html"><img src="https://img.shields.io/badge/Paper-link-important"></a>
	<a href="https://arxiv.org/abs/2307.01026"><img src="https://img.shields.io/badge/arXiv-pdf-yellowgreen"></a>
	<a href="https://pypi.org/project/py-tgb/"><img src="https://img.shields.io/pypi/v/py-tgb.svg?color=brightgreen"></a>
	<a href="https://tgb.complexdatalab.com/"><img src="https://img.shields.io/badge/website-blue"></a>
	<a href="https://docs.tgb.complexdatalab.com/"><img src="https://img.shields.io/badge/docs-orange"></a>
</h3> 

**TGB 2.0: A Benchmark for Learning on Temporal Knowledge Graphs and Heterogeneous Graphs** (NeurIPS 2024 Datasets and Benchmarks Track)
<h3>
  <a href="https://openreview.net/forum?id=EADRzNJFn1#discussion"><img src="https://img.shields.io/badge/Paper-link-important"></a>
  <a href="https://arxiv.org/abs/2406.09639v1"><img src="https://img.shields.io/badge/arXiv-pdf-yellowgreen"></a>
  <a href="https://pypi.org/project/py-tgb/"><img src="https://img.shields.io/pypi/v/py-tgb.svg?color=brightgreen"></a>
	<a href="https://tgb.complexdatalab.com/"><img src="https://img.shields.io/badge/website-blue"></a>
</h3> 


Overview of the Temporal Graph Benchmark (TGB) pipeline:
- TGB includes large-scale and realistic datasets from 10 different domains with both dynamic link prediction and node property prediction tasks.
- TGB automatically downloads datasets and processes them into `numpy`, `PyTorch` and `PyG compatible TemporalData` formats. 
- Novel TG models can be easily evaluated on TGB datasets via reproducible and realistic evaluation protocols. 
- TGB provides public and online leaderboards to track recent developments in temporal graph learning domain.
- Now TGB supports temporal homogeneous graphs, temporal knowledge graphs and temporal heterogenenous graph datasets.

![TGB dataloading and evaluation pipeline](imgs/pipeline.png)

**To submit to [TGB leaderboard](https://tgb.complexdatalab.com/), please fill in this [google form](https://forms.gle/SEsXvN1QHo9tSFwx9)**

**See all version differences and update notes [here](https://tgb.complexdatalab.com/docs/update/)**

### Announcements

**Excited to announce TGB 2.0, has been presented at NeurIPS 2024 Datasets and Benchmarks Track**

See our [camera ready version](https://openreview.net/forum?id=EADRzNJFn1#discussion) and [arXiv version](https://arxiv.org/abs/2307.01026) for details. Please [install locally](https://tgb.complexdatalab.com/docs/home/) first. We welcome your feedback and suggestions. 


**Excited to announce TGX, a companion package for analyzing temporal graphs in WSDM 2024 Demo Track**

TGX supports all TGB datasets and provides numerous temporal graph visualization plots and statistics out of the box. See our paper: [Temporal Graph Analysis with TGX](https://arxiv.org/abs/2402.03651) and [TGX website](https://complexdata-mila.github.io/TGX/).

<!-- **Excited to announce that TGB has been accepted to NeurIPS 2023 Datasets and Benchmarks Track**

Thanks to everyone for your help in improving TGB! we will continue to improve TGB based on your feedback and suggestions.  -->

**Please update to version `2.0.0`**
#### version `2.0.0`

Includes all new datasets from TGB 2.0 including temporal knowledge graphs and temporal heterogeneous graphs. 

<!-- **Please update to version `0.9.2`**

#### version `0.9.2`

Update the fix for `tgbl-flight` where now the unix timestamps are provided directly in the dataset. If you had issues with `tgbl-flight`, please remove `TGB/tgb/datasets/tgbl_flight`and redownload the dataset for a clean install -->


<!-- 
#### version `0.9.1`

Fixed an issue for `tgbl-flight` where the timestamp conversion is incorrect due to time zone differences. If you had issues with `tgbl-flight` before, please update your package. 


#### version `0.9.0`

Added the large `tgbn-token` dataset with 72 million edges to the `nodeproppred` dataset. 

Fixed errors in `tgbl-coin` and `tgbl-flight` where a small set of edges are not sorted chronologically. Please update your dataset version for them to version 2 (will be prompted in terminal). -->


### Pip Install

You can install TGB via [pip](https://pypi.org/project/py-tgb/). **Requires python >= 3.9**
```
pip install py-tgb
```

### Links and Datasets

The project website can be found [here](https://tgb.complexdatalab.com/).

The API documentations can be found [here](https://shenyanghuang.github.io/TGB/).

all dataset download links can be found at [info.py](https://github.com/shenyangHuang/TGB/blob/main/tgb/utils/info.py)

TGB dataloader will also automatically download the dataset as well as the negative samples for the link property prediction datasets.

if website is unaccessible, please use [this link](https://tgb-website.pages.dev/) instead.


### Running Example Methods

- For the dynamic link property prediction task, see the [`examples/linkproppred`](https://github.com/shenyangHuang/TGB/tree/main/examples/linkproppred) folder for example scripts to run TGN, DyRep and EdgeBank on TGB datasets.
- For the dynamic node property prediction task, see the [`examples/nodeproppred`](https://github.com/shenyangHuang/TGB/tree/main/examples/nodeproppred) folder for example scripts to run TGN, DyRep and EdgeBank on TGB datasets.
- For all other baselines, please see the [TGB_Baselines](https://github.com/fpour/TGB_Baselines) repo.

### Acknowledgments
We thank the [OGB](https://ogb.stanford.edu/) team for their support throughout this project and sharing their website code for the construction of [TGB website](https://tgb.complexdatalab.com/).


### Citation

If code or data from this repo is useful for your project, please consider citing our TGB and TGB 2.0 paper:
```
@article{huang2023temporal,
  title={Temporal graph benchmark for machine learning on temporal graphs},
  author={Huang, Shenyang and Poursafaei, Farimah and Danovitch, Jacob and Fey, Matthias and Hu, Weihua and Rossi, Emanuele and Leskovec, Jure and Bronstein, Michael and Rabusseau, Guillaume and Rabbany, Reihaneh},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```

```
@article{huang2024tgb2,
  title={TGB 2.0: A Benchmark for Learning on Temporal Knowledge Graphs and Heterogeneous Graphs},
  author={Gastinger, Julia and Huang, Shenyang and Galkin, Mikhail and Loghmani, Erfan and Parviz, Ali and Poursafaei, Farimah and Danovitch, Jacob and Rossi, Emanuele and Koutis, Ioannis and Stuckenschmidt, Heiner and      Rabbany, Reihaneh and Rabusseau, Guillaume},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```

<!-- 

### Install dependency
Our implementation works with python >= 3.9 and can be installed as follows

1. set up virtual environment (conda should work as well)
```
python -m venv ~/tgb_env/
source ~/tgb_env/bin/activate
```

2. install external packages
```
pip install pandas==1.5.3
pip install matplotlib==3.7.1
pip install clint==0.5.1
```

install Pytorch and PyG dependencies (needed to run the examples)
```
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

3. install local dependencies under root directory `/TGB`
```
pip install -e .
```


### Instruction for tracking new documentation and running mkdocs locally

1. first run the mkdocs server locally in your terminal 
```
mkdocs serve
```

2. go to the local hosted web address similar to
```
[14:18:13] Browser connected: http://127.0.0.1:8000/
```

Example: to track documentation of a new hi.py file in tgb/edgeregression/hi.py


3. create docs/api/tgb.hi.md and add the following
```
# `tgb.edgeregression`

::: tgb.edgeregression.hi
```

4. edit mkdocs.yml 
```
nav:
  - Overview: index.md
  - About: about.md
  - API:
	other *.md files 
	- tgb.edgeregression: api/tgb.hi.md
```

### Creating new branch ###
```
git fetch origin

git checkout -b test origin/test
```

### dependencies for mkdocs (documentation)
```
pip install mkdocs
pip install mkdocs-material
pip install mkdocstrings-python
pip install mkdocs-jupyter
pip install notebook
```


### full dependency list
Our implementation works with python >= 3.9 and has the following dependencies
```
pytorch == 2.0.0
torch-geometric == 2.3.0
torch-scatter==2.1.1
torch-sparse==0.6.17
torch-spline-conv==1.2.2
pandas==1.5.3
clint==0.5.1
``` -->
