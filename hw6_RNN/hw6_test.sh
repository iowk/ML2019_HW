#!/bin/bash
wget 'https://www.dropbox.com/s/2vzaz7v2i0qn4ny/w2v_model.h5'
wget 'https://www.dropbox.com/s/e1dvj5llbjzkxyr/w2v.model.wv.vectors.npy'
wget 'https://www.dropbox.com/s/kxe1l4jej52dvas/w2v.model.trainables.syn1neg.npy'
wget 'https://www.dropbox.com/s/zylb5gjpoq7a2n2/w2v.model'
python3 rf_seg_list_test.py $1 $2
python3 test_w2v.py $3