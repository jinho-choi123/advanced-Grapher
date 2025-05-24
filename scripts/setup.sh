#!/usr/bin/env bash

# install corpusreader
git clone https://gitlab.com/webnlg/corpus-reader.git corpusreader

# install WebNLG_Text_to_triples
git clone https://github.com/WebNLG/WebNLG-Text-to-triples.git WebNLG_Text_to_triples

# install dataset
wget https://github.com/jinho-choi123/advanced-Grapher/raw/refs/heads/artifact-evaluation/webnlg-dataset.zip?download= -O webnlg-dataset.zip
unzip webnlg-dataset.zip


# install pretrained checkpoints
wget https://github.com/jinho-choi123/advanced-Grapher/raw/refs/heads/artifact-evaluation/output.zip?download= -O output.zip
unzip output.zip

# install model config cache
wget https://github.com/jinho-choi123/advanced-Grapher/raw/refs/heads/artifact-evaluation/cache.zip?download= -O cache.zip
unzip cache.zip


# install conda dependencies
conda env create -q -f requirements.yml -n advanced_grapher_env
