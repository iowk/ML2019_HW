#!/bin/bash
python3 rf_seg_list_test.py $3 $4
python3 rf_seg_list.py $1 $4
python3 rf_y_train.py $2
python3 train_w2v.py 