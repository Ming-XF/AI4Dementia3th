#!/bin/bash


python main.py --model "VIB" --num_repeat 3 --dataset 'C42B' --data_dir "../data/C42B/C42B128.npy" --percentage 1. --batch_size 16 --num_epochs 200 --drop_last False --integration "add" --cor_comput "pearson" --d_model 64 --window_size 50 --window_stride 3 --dynamic_length 440 --abla_channel -1 --abla_vae "n" --num_layers 1 --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test
