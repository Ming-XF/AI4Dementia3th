#!/bin/bash

python main.py --model "TCACNet" --num_repeat 3 --dataset 'Beirut' --data_dir "../data/Beirut/Beirut.npy" --batch_size 16 --num_epochs 200 --drop_last False --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test

