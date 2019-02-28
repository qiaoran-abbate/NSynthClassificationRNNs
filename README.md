---
# CONTENTS OF EACH FOLDER

---
* net_model1.pth	 	 : Saved model parameters of model1.py
* model1.py 	  	 : LSTM Model to classify single note recordings of instruments into one of 10 instrument family names 


---
* net_model2.pth	 	 : Saved model parameters of model2.py
* model2.py 	  	 : BLST Model to classify single note recordings of instruments into one of 10 instrument family names 


---
* modelhelper.py	  	 : Helper .py file which contains all the common methods required for model1 and model2


---
# HOW TO RUN/TEST THE MODELS

* python3 [model1.py]

* Do you want to test the model? (1-Yes, 0-No): [Enter 1, to test the model. If 0, then the model will train]

* Enter the model file path: [Give the .pth file path]

---

# Overview
Previously, we have explored feedforward neural network\fully connected network and convolutional network on the NSynth dataset. With fully connect model, we achieved a result of 45.752%. The convolutional network achieved an accuracy of 52.515%. This time, we will continue to explore the NSynth dataset, with yet another type of the neural network models - RNNs. Specifically, we will look into how LSTM and BLSTM perform on this dataset.
Section Training will contain the graphical representations of the network architectures used in both models, and the hyperparameters used to achieve the best result. Section Result described the training result both in a textual and graphical manner. Lastly, Section Discussion will conclude our findings and thoughts on the project itself.

# Training
## Data Processing 
This model uses a regular LSTM model which takes the first second of all NSynth dataset, each of which contains 160000 music frequencies. The batch size of this model is currently set to 32, let’s just look into the 1 sample from the epoch. The sample has sample has the shape of 1*16000 across 1 second. We would like to have 1* 160 samples per time step, which means there will be 16000/160 = 100-time steps in the model. Assuming we want our input size is 100, depending on whether the network has the batch first setting on or not. The input tensor will process the following shape:

# 
    Batch first = T
    Input(batch_size * sequence_length * input_size) = Input (32 * 100 * 160)
#
# 
    Batch first = F
    Input(sequence_length * batch_size * input_size) = Input(100 * 32 * 160)
#

## Training Setup
This section contains the details regarding network architectures and hyperparameters used to achieve the best result. See the following table for comparison.



