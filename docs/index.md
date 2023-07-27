# Welcome to Temporal Graph Benchmark
![TGB logo](assets/logo.png)



### Pip Install

You can install TGB via [pip](https://pypi.org/project/py-tgb/)
```
pip install py-tgb
```

### Links and Datasets

The project website can be found [here](https://tgb.complexdatalab.com/).

The API documentations can be found [here](https://shenyanghuang.github.io/TGB/).

all dataset download links can be found at [info.py](https://github.com/shenyangHuang/TGB/blob/main/tgb/utils/info.py)

TGB dataloader will also automatically download the dataset as well as the negative samples for the link property prediction datasets.


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
```



<!-- ## Code blocks

`pip install tgb` -->



<!-- 

### Plain codeblock

A plain codeblock:

```
Some code here
def myfunction()
// some comment
```

#### Code for a specific language

Some more code with the `py` at the start:

``` py
import tensorflow as tf
def whatever()
```

#### With a title

``` py title="bubble_sort.py"
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```

#### With line numbers

``` py linenums="1"
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```

#### Highlighting lines

``` py hl_lines="2 3"
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```

## Icons and Emojs

:smile: 

:fontawesome-regular-face-laugh-wink:

:fontawesome-brands-twitter:{ .twitter }

:octicons-heart-fill-24:{ .heart } -->