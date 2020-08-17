# Twitter-sentiment-analysis
Downstreaming ULMFiT for the purpose of tweet sentiment analysis(a classification problem).

[Link to the colab notebook.](https://colab.research.google.com/drive/1ijcUtw5eQP66Mu4af_x6eNiVpEl8dcUM#scrollTo=9juKzwz8dw3q)

## Dataset
The dataset comes from kaggle([link](https://www.kaggle.com/kazanova/sentiment140)) that contains 1.6 million tweets and each being labeled as negative(0) or positive(4). The dataset has been sanitized by removing the emoji and emoticons.

## ULMFiT
This is a language model created by **fast.ai** and can be used to downstream for various problems. Our problem today is a classficiation problem. This model is provided via fastai library in python that is build upon pytourch framework. The model can be trained within hours to fit a certain surpervised problem. The fastai library also comes with a **Tokenizer** and **Numericalizer** that will convert the text into appropriate tokens that can then be fed into the Neural Network.
### Inductive transfer learning
**Transfer learning** is a method by which models can transfer the knowledge acquired while training for one task(source task) to be used while training and predicting for another task(called target task). For transfer learning to become **Inductive transfer learning** it has to match two criteria: (1) the source task has to be different from the target task; (2)labeled data has to available in the target domain. As the ULMFiT model was trained as a launage model and our target is to process tweets for sentiment analysis, we are able to fit the criteria very well.
### Architecture of the model
The model can be divided into three parts:
* **General-Domain LM Pretraining:** In a first step, a LM is pretrained on a large general-domain corpus. Now, the model is able to predict the next word in a sequence (with a certain degree of certainty). Figuratively speaking, at this stage the model learns the general features of the language.
* **Target Task LM Fine-Tuning:** Following the transfer learning approach, the knowledge gained in the first step should be utilized for the target task. As the dataset for training the langauge model and target task are different fine-tuning is done.
* **Target Task Classifier:** In this step, the model is used for sentiment analysis, this is done by expanding the model with extra layers.
The model contains a layer of embedding for the purpose of **word embedding** with 400 demensions of vector space, an embedding also converts a word into a numerical valued feature set. This embedding helps words to be closely related to similar meaning words and is better than one-hot encoding as this is denser.
### Freezing
As their is a change in the embedding layer after **General-Domain LM Pretraining** to **Target Task LM Fine-tuning** we freeze the model except the embedding layer and the decoding layer, so as to not forget everything the model has learned so far. This freezing is done for 1 epoch and then the rest of the model is unfreezed for fine-tuning.
### Learning Rate Schedule
This is a method to get the best learning rate. In this the learning rate is not kept constant rather it keeps on increasing from shoter steps(or smaller values of learning rate) to larger steps(or larger values of learning rate) to converge in the parameters space.
### Gradual unfreezing
As for the classifier we add two linear models that have been initialized with random numbers. If the entire model is trained right now then the model can forget everything it has learned. To prevent this forgottening the model is freezed except the newly added layers. This newly added layers are trained first but this is done by freezing the model upto second last layer and fine-tuned for 1 epoch and then again fine-tuned while freezing upto newly added layers for 1 epoch. After this rest of the model is unfrozen to be fine-tuned.
## Benchmarks
The model is able to reach upto 82.2% accuracy.
