# Pick and Choose Graph Neural Network implementation

PyTorch Pick and Choose implementation for baseline comparison purposes. You can find the original paper [here](https://ponderly.github.io/pub/PCGNN_WWW2021.pdf)

## Setup

``` bash
python3 -m venv .env
. .env/bin/activate
pip install -r requirements.txt
pip install -e .
```

## How to execute

``` bash
python3 pac_gnn/main.py
```

If you use this code in your work, please cite my paper that implements it:

```
@article{DBLP:journals/corr/abs-2105-14568,
  author    = {Ronald D. R. Pereira and
               Fabricio Murai},
  title     = {How effective are Graph Neural Networks in Fraud Detection for Network
               Data?},
  journal   = {CoRR},
  volume    = {abs/2105.14568},
  year      = {2021},
  url       = {https://arxiv.org/abs/2105.14568},
  archivePrefix = {arXiv},
  eprint    = {2105.14568},
  timestamp = {Wed, 02 Jun 2021 11:46:42 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2105-14568.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
