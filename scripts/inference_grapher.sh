#!/usr/bin/env bash

# FIXME: for fast testing, limit_test_batches set to 0.1, at real-evaluation we should recover the value to 1.0
python main.py    --version 1\
                  --default_root_dir output \
                  --run inference \
                  --max_epochs 1 \
                  --accelerator gpu \
                  --num_nodes 1 \
                  --devices "0," \
                  --batch_size 120 \
                  --num_sanity_val_steps 10 \
                  --fast_dev_run 0 \
                  --overfit_batches 0 \
                  --limit_train_batches 1.0 \
                  --limit_val_batches 1.0 \
                  --limit_test_batches 1.0 \
                  --accumulate_grad_batches 1 \
                  --detect_anomaly True \
                  --data_path webnlg-dataset/release_v3.0/en \
                  --checkpoint_model_id 79 \
                  --check_val_every_n_epoch 3 \
                  --inference_input_text "Danielle Harris had a main role in Super Capers, a 98 minute long movie."


                  # set add_rgcn flag if you want to test with rgcn added
                  # if add_rgcn flag is set, we should reduce batch_size
