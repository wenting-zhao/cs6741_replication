# CS6741 Replication
Replication project for CS6741 where I reproduced https://arxiv.org/abs/1908.04626

# How to run the code
To run baseline:
```CUDA_VISIBLE_DEIVICES=0 python main.py --dataset {AgNews, sst, imdb, 20News_sports} --data_dir . --output_dir test_outputs/ --attention tanh --encoder lstm --epoch 8 --batch_size 32 --lr 0.001 --use_attention```

To run uniform as the adversary:
```CUDA_VISIBLE_DEIVICES=0 python main.py --dataset {AgNews, sst, imdb, 20News_sports} --data_dir . --output_dir test_outputs/ --attention frozen --encoder lstm --epoch 8 --batch_size 32 --lr 0.001 --use_attention```
