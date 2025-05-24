<div align="center">

# two-trial Text Knowledge Graph Generation(TeKGG)

## Description
2-trial Text Knowledge Graph Generation(TeKGG) suggest a novel method of generating Knowledge Graph from text.
1. Generate first-trial Knowledge Graph using [Grapher](https://github.com/IBM/Grapher). Extract the Knowledge Graph and node embeddings.
2. Using the extracted node embeddings and first-trial Knowledge Graph, we apply GCN message passing to aggregate node embeddings from neighbors.
3. Using newly generated node embeddings, it is again passed to Grapher, and generate second-trial Knowledge Graph.

</div>

## Important
For `git clone` command, disable fetching files from LFS. Due to bandwidth/storage limit, you cannot fully get the content.
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:jinho-choi123/advanced-Grapher.git
```

## Running in Colab
You can easily run this project in Colab with [`main.ipynb`](main.ipynb).

## Running in your Own Machine


### Environment Setup
We use conda to setup environment. Running `scripts/setup.sh` would be enough.
This script will setup all the conda environments, datasets, and other 3rd-party dependencies.
```bash
sh scripts/setup.sh
```

We use WebNLG 3.0 dataset from [here](https://gitlab.com/shimorina/webnlg-dataset.git)

### How to train
We have two training phase:
1. Training Grapher model
2. Training GCN Model

```bash
# Train Grapher model first
sh scripts/train_grapher.sh

# Train GCN model next
sh scripts/train_rgcn.sh
```

### How to test
We already contained pretrained checkpoints at `output/webnlg_version_1/checkpoints/model-epoch=79.ckpt` and `output/webnlg_version_1/checkpoints/model-epoch=114.ckpt`.
The script is initially configured to point to these pretrained checkpoints.
```bash
# Test the grapher only model
sh scripts/test_grapher.sh

# Test the advanced-grapher model
sh scripts/test_advanced_grapher.sh
```

### How to run inference
You can run single graph generation inference using following command. If you want to change the text, please modify `--inference_input_text` argument of the script.
```bash
# Inference the grapher-only model.
sh scripts/inference_grapher.sh

# Inference the advanced-grapher model.
sh scripts/inference_advanced_grapher.sh
```

### Results
Results can be visualized in Tensorboard
```bash
tensorboard --logdir output
```

### References
```
@inproceedings{grapher2022,
  title={Knowledge Graph Generation From Text},
  author={Igor Melnyk, Pierre Dognin, Payel Das},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (Findings of EMNLP)},
  year={2022}
}
```
