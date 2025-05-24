#!/usr/bin/env bash

# install conda dependencies
conda env create -q -f requirements.yml -n advanced_grapher_env
conda init
conda activate advanced_grapher_env

git clone https://gitlab.com/webnlg/corpus-reader.git corpusreader
