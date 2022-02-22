# MetaHG
Distilling Meta Knowledge on Heterogeneous Graph for Illicit Drug Trafficker Detection on Social Media
====
Official source code of "Distilling Meta Knowledge on Heterogeneous Graph for Illicit Drug Trafficker Detection on Social Media
" (NeurIPS 2021 https://proceedings.neurips.cc/paper/2021/file/e234e195f3789f05483378c397db1cb5-Paper.pdf).

## Requirements

This code is developed and tested with python 3.7.6. and requires the following:

* dgl==0.7.2+cu111
* torch==1.8.1+cu111
* tqdm==4.23.4
* sklearn==0.24.2
* numpy==1.16.5
* pandas==0.25.1
* json==2.0.9

run `pip install -r requirements.txt` to install all the dependencies. 

## Usage

This source code contains two parts, representation learning and meta learning.

Representation Learning:

```bg_gsr.py``` contains the code of graph construction and graph structure refinement. 

```rgcn_mode.py``` includes the code of rgcn model for node represenation learning.

```ssl.py``` has the code of self-supervised learning based on the similarities among nodes.

Meta Learning:

```meta.py``` contains the code of meta-learning. 

```learner.py``` includes the code of meta learner (two-layer MLP in this paper) in meta-learning.


We can run the code:

```python main.py``` 

to train the model which integrates GSR (graph structure refinement), SSL (self-supervised Learning), R-GCNs (relation graph convolutional network), and Meta (Meta-Learning).


## Contact
Yiyue Qian - yqian5@nd.edu or yxq250@case.edu

Discussions, suggestions and questions are always welcome!

Due to privacy issues, all data we collected and utilized for this paper is not able to be publicly accessed for the time being. If you have interests in our data for research purposes, please send emails to me for further details.


## Citation
```
@inproceedings{qian2021distilling,
  title={Distilling Meta Knowledge on Heterogeneous Graph for Illicit Drug Trafficker Detection on Social Media},
  author={Qian, Yiyue and Zhang, Yiming and Ye, Yanfang and Zhang, Chuxu},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

