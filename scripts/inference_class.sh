#!/usr/bin/env bash

# FIXME: for fast testing, limit_test_batches set to 0.1, at real-evaluation we should recover the value to 1.0
python main.py    --version 1\
                  --default_root_dir output \
                  --run inference \
                  --max_epochs 50 \
                  --accelerator gpu \
                  --num_nodes 1 \
                  --devices "0," \
                  --num_data_workers 32 \
                  --lr 1e-4 \
                  --batch_size 50 \
                  --num_sanity_val_steps 0 \
                  --fast_dev_run 0 \
                  --overfit_batches 0 \
                  --limit_train_batches 1.0 \
                  --limit_val_batches 1.0 \
                  --limit_test_batches 1.0 \
                  --accumulate_grad_batches 5 \
                  --detect_anomaly True \
                  --data_path webnlg-dataset/release_v3.0/en \
                  --log_every_n_steps 100 \
                  --val_check_interval 1000 \
                  --checkpoint_step_frequency 1000 \
                  --focal_loss_gamma 3 \
                  --dropout_rate 0.5 \
                  --num_layers 2 \
                  --edges_as_classes 1 \
                  --checkpoint_grapher_model_id -1 \
                  --checkpoint_rgcn_model_id -1 \
                  --precision "bf16" \
                  --inference_input_text "Danielle Harris had a main role in Super Capers, a 98 minute long movie."
                  # --add-rgcn  \


                  # set add_rgcn flag if you want to test with rgcn added
                  # if add_rgcn flag is set, we should reduce batch_size
