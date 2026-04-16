#!/bin/bash

python main.py --model "SteadyNet" --num_repeat 3 --dataset 'C42B' --data_dir "../data/C42B/C42B128.npy" --batch_size 16 --num_epochs 200 --num_kernels 10 --drop_last False --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test
