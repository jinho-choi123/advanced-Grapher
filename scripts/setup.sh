#!/usr/bin/env bash

git clone https://gitlab.com/webnlg/corpus-reader.git corpusreader

wget https://github.com/jinho-choi123/advanced-Grapher/raw/refs/heads/artifact-evaluation/webnlg-dataset.zip?download= -O webnlg-dataset.zip
unzip webnlg-dataset.zip

wget https://github.com/jinho-choi123/advanced-Grapher/raw/refs/heads/artifact-evaluation/output.zip?download= -O output.zip
unzip output.zip

wget https://github.com/jinho-choi123/advanced-Grapher/raw/refs/heads/artifact-evaluation/cache.zip?download= -O cache.zip
unzip cache.zip


# install conda dependencies
conda env create -q -f requirements.yml -n advanced_grapher_env
conda init
conda activate advanced_grapher_env
