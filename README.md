# RoT: Enhancing Table Reasoning with Iterative Row-Wise Traversals

This repository contains code for the paper ["RoT: Enhancing Table Reasoning with Iterative Row-Wise Traversals"](https://aclanthology.org/2025.emnlp-main.29/).

If you use RoT in your work, please cite it as follows:
```
@inproceedings{zhang-etal-2025-rot,
    title = "{R}o{T}: Enhancing Table Reasoning with Iterative Row-Wise Traversals",
    author = "Zhang, Xuanliang  and
      Wang, Dingzirui  and
      Xu, Keyan  and
      Zhu, Qingfu  and
      Che, Wanxiang",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.29/",
    pages = "559--579",
    ISBN = "979-8-89176-332-6"
}
```

## Build Environment
```
conda create -n rot python=3.10
conda activate rot
pip install -r requirements.txt
```

## Pre-Process Data
Download and put each dataset in ./dataset/[dataset_name]/raw, and run [dataset/slurm/preprocess.slurm](./dataset/slurm/preprocess.slurm).

## Download Model
Download the models and put them in ./model. Write your config in ./config.

## RoT
Run the table reasoning with [inference/slurm/inference_traverse.sh](./inference/slurm/inference_traverse.sh).
