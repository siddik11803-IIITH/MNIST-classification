# MNIST Classification Project

The Following repository contains the code for the MNIST Dataset Classification Problem. The repostiory is made of Jupyter Notebooks and Python Code aiming for the same thing - MNIST classification problem.

The MNIST dataset is a hand labelled dataset consisitng of images of handwrtiten digits written by high school students and employees of the United states Census Bureau. The dataset consists of images of digits ranging from 0 to 9. 

[Wikipedia Page](https://en.wikipedia.org/wiki/MNIST_database)

![MNIST Dataset](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

The Accuracies achieved in the codes are as follows. (The report can be found in [nn.pdf](https://github.com/siddik11803-IIITH/MNIST-classification/blob/main/nn.pdf))

### DNN 
- 5 Epochs
- 1 Hidden Layer - 200 Neurons
Parameters/Set | Loss | Accuracy
--- | --- | --- 
Training Set | 0.16881218552589417 | 0.9567333459854126
Test Set | 0.24324145913124084 | 0.9473000168800354

### CNN
- 5 Epochs
- 3 Hidden Layers - (32, 64, 1024 respectively)
Parameters/Set | Loss | Accuracy
--- | --- | --- 
Training Set | 0.056159235537052155 | 0.9822666645050049
Test Set | 0.05705214664340019 | 0.9821000099182129

APIs Written in Backend:
1. DNN &#8594; Finished
2. CNN &#8594; Finished
3. K-Means &#8594; Under Work
4. GMM &#8594; Under Work