# CS6741 Replication
Replication project for CS6741 where I reproduced https://arxiv.org/abs/1908.04626

# How to run the code
To run baseline:
```CUDA_VISIBLE_DEIVICES=0 python main.py --dataset {AgNews, sst, imdb, 20News_sports} --data_dir . --output_dir test_outputs/ --attention tanh --encoder lstm --epoch 8 --batch_size 32 --lr 0.001 --use_attention```

To run uniform as the adversary:
```CUDA_VISIBLE_DEIVICES=0 python main.py --dataset {AgNews, sst, imdb, 20News_sports} --data_dir . --output_dir test_outputs/ --attention frozen --encoder lstm --epoch 8 --batch_size 32 --lr 0.001 --use_attention```

# Introduction
In this project, we replicate parts of the main experiments in __Attention is not not Explanation__ from scratch. In particular, we reproduce the baseline experiment where Wiegreffe et al. (2019) applied a single-layer bidirectional LSTM with tanh activation, followed by an additive attention layer and sigmoid prediction for binary classification. Additionally, we implemented the __Uniform as the Adversary__ experiment. We obtained slightly different results that counter the original claim made in the paper. Finally, we partially implement the diagnose model (adversarial training is in progress).

# Method
## Datasets
Although there are six datasets included in the experiment, we only have easy access to four of them.  They are AgNews, 20News_sports, SST, and IMDb. We didn't process the datasets by ourselves. The preprocess code can be found at https://github.com/successar/AttentionExplanation/tree/master/preprocess. We use https://github.com/sarahwie/attention/blob/7dfda3d232ae40af40e5f5ffe7c9585ee1713735/Trainers/DatasetBC.py to read the data.

## Training and Test
To build up the experiment framework, I have the following files:

  main.py - get the whole experiment to start, where it calls
  
    config_args.py - read and process the command line arguments
    
    model/Models.py - initialize the model that consists of
    
      model/Attention.py - implements the attention mechanism
      
      model/Encoders.py - implements the encoders
      
      model/Decoders.py - implements the decoders
      
    runner.py - handles training and testing
    
      train.py - runs the training code
      
      test.py - runs the testing code
      
## Notes
To reproduce their performance, I did the following
  1. Have a separate optimizer for the attention parameters, where there is no weight decaying
  2. Instead of doing BCE loss, place a larger weight on class 1
  3. Load pre-trained embedding for the vocabulary

# Results
We include F1 scores from our experiments as well as the results from the orginal papers. We can see that in all cases we outperform (or is comparable to) the original F1 scores.
|        |   Ours   |         |   Theirs   |         |
|:------:|:--------:|:-------:|:----------:|:-------:|
|        | Baseline | Uniform | Baseline   | Uniform |
|  IMDb  |  0.9065  |  0.8861 |    0.902   |  0.879  |
|   SST  |  0.8304  |  0.8248 |    0.831   |  0.822  |
| AgNews |  0.9604  |  0.9595 |    0.964   |  0.960  |
| 20News |  0.9394  |  0.9449 |    0.942   |  0.934  |
