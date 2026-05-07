#!/bin/bash

python main.py --model "CVIB4LMDA" --num_repeat 3 --dataset 'Dementia400' --data_dir "../data/Dementia400/Dementia400.npy"  --batch_size 8 --num_epochs 200 --drop_last False --d_model 64 --abla_channel -1 --abla_vae "n" --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test
