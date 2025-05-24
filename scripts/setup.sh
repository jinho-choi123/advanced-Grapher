#!/usr/bin/env bash

# install conda dependencies
conda env create -f requirements.yml -n advanced_grapher_env
conda activate advanced_grapher_env

git clone https://gitlab.com/webnlg/corpus-reader.git corpusreader
git clone https://gitlab.com/shimorina/webnlg-dataset.git

# copy checkpoints to output/webnlg_version_1/checkpoints/ directory
cp checkpoints/train-grapher-last.ckpt output/webnlg_version_1/checkpoints/model-epoch=79.ckpt
cp checkpoints/train-rgcn-last.ckpt output/webnlg_version_1/checkpoints/model-epoch=114.ckpt
