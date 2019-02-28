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

<table class="tg">
  <tr>
    <th class="tg-oud0"></th>
    <th class="tg-35uj">LSTM</th>
    <th class="tg-35uj">BLSTM</th>
  </tr>
  <tr>
    <td class="tg-fymr">Parameters used</td>
    <td class="tg-0pky">LEARN_RATE = 0.0008<br>BATCH_SIZE = 32<br>EPOCHS = 20<br># Network Parameters<br>INPUT_SIZE = 160<br>HIDDEN_SIZE = 128<br>NUM_LAYERS = 3</td>
    <td class="tg-0pky">LEARN_RATE = 0.0008<br>BATCH_SIZE = 32<br>EPOCHS = 20<br># Network Parameters<br>INPUT_SIZE = 160<br>HIDDEN_SIZE = 128<br>NUM_LAYERS = 3<br>bidirectional = True</td>
  </tr>
  <tr>
    <td class="tg-fymr" rowspan="2">Network architecture</td>
    <td class="tg-0pky">TEST1</td>
    <td class="tg-0pky">TEST2</td>
  </tr>
  <tr>
    <td class="tg-0pky" colspan="2">The RNN network architectures shown above are heavily inspired by Zhou’s Pytorch Tutorial<br>sites (Zhou, 2018).</td>
  </tr>
</table>


# Discussion 

## What does the visualization tell us about the behavior of the LSTM and BLSTM classifiers, and their ability to discriminate between the instrument families?
Both LSTM and BLSTM are able to classify instrument families with lesser variety pretty well, such as String, Vocal, and so on. However, they struggle with similar looking samples, such as keyboard and brass. BLSTM especially struggle with samples that might have a symmetrical shape. The keyboard class is one such example, since the ending of it looks similar, in some case, as the beginning of a bass sample.

## How do the LSTM and BLSTM compare in terms of speed and accuracy compared to your best network from project1?
BLSTM is a lot slower compared to LSTM, due to the additional complexity. BLSTM achieved 63.696%, which is about 1% less than LSTM, that achieved 64.844%. The different is not significant but does prove the point that in machine learning, or a less complicated model sometimes performs better than a complex model. The right fit is the key.

## What in your RNN network designs seems to have led to the results obtained? How might they be changed to improve the results?
Having an Adam optimizer in the gradient descent process definitely helped the model to learn better. Also, adding additional layers within the RNNs (from 2 layers to 3 layers) helped to improve the accuracy for about 3%. We read online that Tanh activation function is often used in RNNs, but adding it to the model did not change the result much. The loss over epoch figure does suggest that the model had overfitted, so given the opportunity, we’d like to adopt weight decay, bigger batch size to improve it. Also, having the entire 4 seconds as sample might yield a better result.

## Additional Comments/Summary
In this project, we had to first familiarize ourselves with the RNNs, and how to reshape the tensors to match the network. Then we tweaked the parameters to improve its accuracy. Notably, changing the optimizer for gradient descent from SGD to Adam made the biggest difference. Which can be explained by the fact that Adam optimizer allows for a dynamic momentum in proportion to the gradient magnitude. Hence, when a model had steep slopes, Adam optimizer helps the model to descend further into the valley, aka the local or global minima.

Overall this project allowed us to have a better understanding of how RNNs such as LSTM and BLSTM work both theoretically and practically. When compared to project 1, the model did achieve a better accuracy. It theoretically makes sense since music notes are time-dependent data, naturally, RNNs should perform better. However, others have achieved similar or better accuracy in their convolutional models. It is possible that we haven’t explored all the best parameter to really optimize the LSTM or the BLSTM models. Not to mention, we are currently studying 1 second of the audio file rather than 4. In the future, we’d like to look at increasing the number of layers and incorporate more seconds of the music notes.
