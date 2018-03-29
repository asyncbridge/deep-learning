# CNN for Sentence Classification
I tried to train/evaluate the Movie Review data using with CNN for Sentence Classification.  

Additionally, I added multi-channel training code and loading code from word2vec.  
The word2vec file should be copied to ./data/word2vec/ path and the file name is "GoogleNews-vectors-negative300.bin".  

The original source is from https://github.com/dennybritz/cnn-text-classification-tf/  

![Alt text](https://github.com/asyncbridge/deep-learning/blob/master/CNNSentenceClassification/CNN_sentence_classification.png?raw=true)  
  
I used Tensorboard and estimate the accuracy of this model as follows.  				  
  
![Alt text](https://github.com/asyncbridge/deep-learning/blob/master/CNNSentenceClassification/CNN_sentence_tensorboard.png?raw=true)  

## Development Environment
__Prerequisite__: Python >= 3.6.3, Tensorflow >= r1.3.0  

## Review
My paper review is as follows.    
  
http://arclab.tistory.com/149  

It is based on a paper and the other references as below.  

## References
[1] https://arxiv.org/abs/1408.5882/  
[2] http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/  
[3] https://github.com/dennybritz/cnn-text-classification-tf/  
[4] https://github.com/yoonkim/CNN_sentence/  
[5] https://github.com/cahya-wirawan/cnn-text-classification-tf/  
[6] https://github.com/harvardnlp/sent-conv-torch/  
[7] https://github.com/mmihaltz/word2vec-GoogleNews-vectors/