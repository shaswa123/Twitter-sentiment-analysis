# Twitter-sentiment-analysis
Downstreaming ULMFiT for the purpose of tweet sentiment analysis(a classification problem).

[Link to the colab notebook.](https://colab.research.google.com/drive/1ijcUtw5eQP66Mu4af_x6eNiVpEl8dcUM#scrollTo=9juKzwz8dw3q)

## Dataset
The dataset comes from kaggle([link](https://www.kaggle.com/kazanova/sentiment140)) that contains 1.6 million tweets and each being labeled as negative(0) or positive(4). The dataset has been sanitized by removing the emoji and emoticons.

## ULMFiT
This is a language model created by **fast.ai** and can be used to downstream for various problems. Our problem today is a classficiation problem. This model is provided via fastai library in python that is build upon pytourch framework. The model can be trained within hours to fit a certain surpervised problem. The fastai library also comes with a **Tokenizer** and **Numericalizer** that will convert the text into appropriate tokens that can then be fed into the Neural Network.
