# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:52:39 2018

@author: Parinitha Nagaraja and Qiaoran Li 
"""
import os
import matplotlib
matplotlib.use('pdf')    
import torch
torch.backends.cudnn.enabled = False
import copy 
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from pytorch_nsynth.nsynth import NSynth
from sklearn import metrics

#Using CUDA if we have a GPU that supports it along with the correct
#Install, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running model 1 on', device)

RESULT_DIR = '_result/'
RESULT_DIR_BEST = '_result/model_result_best/'
RESULT_DIR_LAST = '_result/model_result_last/'
MOMENTUM = 0.8
LEARN_RATE = 0.0008
BATCH_SIZE = 32
EPOCHS = 1
VALIDATION_SPLIT = 0.2
TESTING_SPLIT = 0.1
RAND_SEED = 0  
INSTRUMENT_FAMILY_CLASSES = ['bass','brass','flute','guitar','keyboard','mallet','organ','reed','string','vocal']
toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)


def divide_datasets(RUN_ON_SERVER, want_to_test):  
    # use instrument_family and instrument_source as classification targets 
    if (RUN_ON_SERVER):
        return loadDataForServer()
    else:
        return loadDataForLocal(want_to_test)

def loadDataForServer():
    
    training_dataset = NSynth(
        "/local/sandbox/nsynth/nsynth-train",
        transform=toFloat,
        blacklist_pattern=["synth_lead"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"])
    
    validation_dataset = NSynth(
        "/local/sandbox/nsynth/nsynth-valid",
        transform=toFloat,
        blacklist_pattern=["synth_lead"],  # blacklist synth_lead instrument
        categorical_field_list=["instrument_family", "instrument_source"])
    
    testing_dataset = NSynth(
        "/local/sandbox/nsynth/nsynth-test",
        transform=toFloat,
        blacklist_pattern=["synth_lead"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"])
    
    # create dataLoaders
    train_loader = torch.utils.data.DataLoader(dataset=training_dataset, 
                    batch_size=BATCH_SIZE)
    
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                    batch_size=1)
    
    test_loader = torch.utils.data.DataLoader(dataset=testing_dataset, 
                    batch_size=1)
    
    print('Finished preparing data loaders for server testing')
    
    return train_loader, validation_loader, test_loader


def loadDataForLocal(want_to_test):
    
    training_dataset = NSynth(
        "./nsynth-test",
        transform=toFloat,
        blacklist_pattern=["synth_lead"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"])
            
    # Splitting training dataset into training and validation and testing
    num_train = len(training_dataset)
    indices = list(range(num_train))
    splitVal = int(np.floor(VALIDATION_SPLIT * num_train))
    splitTest = int(np.floor(TESTING_SPLIT * num_train)) + splitVal
      
    # Make sure you get same numbers every time when rand_seed = 0
    np.random.seed(seed = RAND_SEED)
    
    # Shuffle the indices
    np.random.shuffle(indices)
    
    
    # Get training set index and validation set index
    validation_idx, test_idx, train_idx = indices[splitVal:], \
                                          indices[splitVal:splitTest], \
                                          indices[splitTest:]
     
    # create samplers
    train_sampler = data_utils.SubsetRandomSampler(train_idx)
    test_sampler = data_utils.SubsetRandomSampler(test_idx)
    validation_sampler = data_utils.SubsetRandomSampler(validation_idx)
    
    # create dataLoaders
    train_loader = torch.utils.data.DataLoader(dataset=training_dataset, 
                    batch_size=BATCH_SIZE, sampler=train_sampler)
    
    validation_loader = torch.utils.data.DataLoader(dataset=training_dataset, 
                    batch_size=1, sampler=validation_sampler)
    
    test_loader = torch.utils.data.DataLoader(dataset=training_dataset, 
                    batch_size=1, sampler=test_sampler)
    
    if want_to_test == '1':
        test_loader = torch.utils.data.DataLoader(dataset=training_dataset, 
                    batch_size=1)
        
    
    print('Finished preparing data loaders for local testing')
    
    return train_loader, validation_loader, test_loader
    
    
def training(net, train_loader, validation_loader, RUN_ON_SERVER): 

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)
    
    # initialize data needed to store loss value 
    saved_train_loss = []
    saved_val_loss = []
    saved_train_accuracy = []
    saved_val_accuracy = []
    models_dict = {}
    
    # iterate through the epochs
    for epoch in range(EPOCHS):
        
        # initializing parameters 
        total = 0
        correct = 0
        accuracy_per_epoc = 0
        running_loss = 0.0          
        
        for i, data in enumerate(train_loader, 0):      
                
            #Get the inputs
            # Get 1 second for all rows
            inputs = data[0].float()[:, 0:16000]
            inputs = np.reshape(inputs,(len(inputs),100,160))
            
            #inputs = data[0].float()[:, 0:16000]
            #inputs = torch.from_numpy(np.moveaxis(inputs.reshape((100, BATCH_SIZE, 160)).data.numpy(), 1, 0))

            
            labels = data[1]
            
            #inputs = inputs.float()
            
            inputs, labels = inputs.to(device), labels.to(device)
      
            #zero the parameter gradients
            optimizer.zero_grad()
               
            #forward + backward + optimizer
            outputs = net(inputs)
    
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() 
            # compute the accuracy
            probabilities, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                
            
        
        # compute the accuracy per epoch
        accuracy_per_epoc = (100 * correct / total) 
        
        # save train loss to collection 
        train_loss = running_loss/ len(train_loader)
        saved_train_loss.append(train_loss)
              
        # call validate function 
        val_loss, val_accuracy = validate(validation_loader, net, optimizer, criterion) 
        # save val loss to collection 
        saved_val_loss.append(val_loss) 
        
        # save accuracy to collections
        saved_train_accuracy.append(accuracy_per_epoc)
        saved_val_accuracy.append(val_accuracy)
        
        # Save the models
        models_dict[epoch] = copy.deepcopy(net)
        
        print('Epoch ' + str(epoch+1) + '\n' + 
              'training loss: ' + str(train_loss)  + '\t\t' + 
              'validation loss: ' + str(val_loss) + '\n' + 
              'training accuracy:' + str(accuracy_per_epoc) + '\t\t' + 
              'validation accuracy:' + str(val_accuracy) + '\n')
          
    print('Finished Training')    
    
    overfitting_pt = compute_overfittingpoint(saved_val_loss)
    
    # Get the best model
    bestnet = None
    if overfitting_pt != -1:
        bestnet = models_dict[overfitting_pt]
    else:
        bestnet = net
    
     # Save the best bet
    if RUN_ON_SERVER:
        if bestnet != None: 
            torch.save(bestnet.state_dict(), RESULT_DIR + 'bestnet_model1.pth')
        torch.save(net.state_dict(), RESULT_DIR + 'net_model1.pth')
    
    printAvgModel(saved_train_loss, saved_val_loss, saved_train_accuracy, 
                  saved_val_accuracy, overfitting_pt)   
    
    return bestnet, net

def compute_overfittingpoint(saved_losses_validation):
    
    for index in range(len(saved_losses_validation)-1):
        prev_loss_validation = saved_losses_validation[index]
        curr_loss_validation = saved_losses_validation[index+1]
        if curr_loss_validation > prev_loss_validation:
             return index
    
    return -1    
      
def validate(validation_loader, net, optimizer, criterion): 

    running_loss = 0.0
    total = 0
    correct = 0
    accuracy_per_epoc = 0

    for i, data in enumerate(validation_loader, 0):      
        
          
        #Get the inputs
        #inputs = data[0].float()[:, 0:16000]
        #inputs = torch.from_numpy(np.moveaxis(inputs.reshape((100, BATCH_SIZE, 160)).data.numpy(), 1, 0))
        inputs = data[0].float()[:, 0:16000]
        inputs = np.reshape(inputs,(len(inputs),100,160))
        
        labels = data[1]
        
        #inputs = inputs.float()
        
        
        inputs, labels = inputs.to(device), labels.to(device)
  
        #zero the parameter gradients
        optimizer.zero_grad()
           
        #forward + backward + optimizer
        outputs = net(inputs)

        loss = criterion(outputs, labels)
            
        running_loss += loss.item()  
        
        # compute the accuracy
        probabilities, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # compute the accuracy per epoch
    accuracy_per_epoc = (100 * correct / total) 
        
    val_loss = running_loss/ len(validation_loader)
    
    return val_loss, accuracy_per_epoc

def printAvgModel(saved_losses_train, saved_losses_val, saved_train_accuracy, 
                  saved_val_accuracy, overfit_x_value):
    
    # convert data to array 
    saved_losses_train = np.array(saved_losses_train)
    saved_losses_val = np.array(saved_losses_val) 
    
    #  display the accuracy graph
    saved_accuracy_train = np.array(saved_train_accuracy)
    saved_accuracy_val = np.array(saved_val_accuracy)
     
    #  display the loss graph
    f, ax = plt.subplots(2, sharex = True)
    ax[0].plot(saved_losses_train, color='blue', marker=".", label = 'train loss')
    ax[0].plot(saved_losses_val, color='red', marker= ".", label = 'val loss')
    ax[0].legend(loc ='upper right')
    ax[0].set_ylabel("Average Loss")
    
    ax[1].plot(saved_accuracy_train, color='blue', marker=".", label = 'train accuracy')
    ax[1].plot(saved_accuracy_val, color='red', marker= ".", label = 'val accuracy')
    ax[1].legend(loc ='upper left')
    ax[1].set_ylabel("Average Accuracy")
    # plt.axvline(x = overfit_x_value)
    
    print("Momentum:", MOMENTUM, " LR:", LEARN_RATE)
    title = 'LR = {}, Momentum = {}, Epochs = {}'.format(LEARN_RATE, MOMENTUM, EPOCHS)
    f.suptitle(title)  
    plt.xlabel("Epochs")
    f.savefig( RESULT_DIR + 'AverageModelLossModel2', bbox_inches='tight')
    plt.close()
    
  

def compute_accuracy(bestmodel,testloader, RUN_ON_SERVER, model_Type):
    
    correct = 0 
    total = 0 
    true_labels = list()
    predicted_labels = list()
    positive_Inf = float("inf")
    negative_Inf = float("-inf")
    
    max_probability = { 0:negative_Inf, 
						1:negative_Inf, 
						2:negative_Inf, 
						3:negative_Inf, 
						4:negative_Inf, 
						5:negative_Inf, 
						6:negative_Inf, 
						7:negative_Inf, 
						8:negative_Inf, 
						9:negative_Inf}
    
    min_diff = {0:positive_Inf, 
			   1:positive_Inf, 
			   2:positive_Inf, 
			   3:positive_Inf, 
			   4:positive_Inf, 
			   5:positive_Inf, 
			   6:positive_Inf, 
			   7:positive_Inf, 
			   8:positive_Inf, 
			   9:positive_Inf}
    
    max_diff = { 0:negative_Inf, 
                1:negative_Inf, 
                2:negative_Inf, 
                3:negative_Inf, 
                4:negative_Inf, 
                5:negative_Inf, 
                6:negative_Inf, 
                7:negative_Inf, 
                8:negative_Inf, 
                9:negative_Inf}
    
    
    min_probability = { 0:positive_Inf, 
						1:positive_Inf, 
						2:positive_Inf, 
						3:positive_Inf, 
						4:positive_Inf, 
						5:positive_Inf, 
						6:positive_Inf, 
						7:positive_Inf, 
						8:positive_Inf, 
						9:positive_Inf}
    
    with torch.no_grad():
        for data in testloader:
#            inputs = data[0].float()[:, 0:16000]
#            inputs = torch.from_numpy(np.moveaxis(inputs.reshape((100, BATCH_SIZE, 160)).data.numpy(), 1, 0))
            
            inputs = data[0].float()[:, 0:16000]
            inputs = np.reshape(inputs,(len(inputs),100,160))
        
            labels = data[1]
            total += labels.size(0)
            
            #inputs = inputs.float()   
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = bestmodel(inputs)
            probability, predicted = torch.max(outputs.data, 1)
            
            # get all probabilities 
            predicted = predicted.item()
            predicted_probability = probability.item()
            label = labels.item()
            label_probability = outputs.data[0][label]


            if RUN_ON_SERVER:
                inputs = inputs[0][0].cpu().numpy()
            else:
                inputs = inputs[0][0].numpy()
            
            
            if predicted == label:
                correct += 1 
				
				# visualize the sound wave
                if predicted_probability > max_probability[label]:
					# overwrite the max prob image
                    saveImage(inputs, label, '_Max', model_Type, predicted_probability)
                    max_probability[label] = predicted_probability
#                else:
#                    if probability > second_max_probability[labels]:
#                        saveImage(inputs, labels, '_2nd_Max_', model_Type)
#                        second_max_probability[labels] = probability
                        
                if predicted_probability < min_probability[label]: 
					# overwrite the min prob image
                    saveImage(inputs, label, '_Min', model_Type, predicted_probability)
                    min_probability[label] = predicted_probability
            else:
                difference = (predicted_probability - label_probability).item()
                title = str(difference) + ' away from the boundary'
                
                # min diff for each class and max diff for each class 
                if difference < min_diff[label]:
                    
                    saveImage(inputs, label, '_Min_Diff', model_Type, title)
                    min_diff[label] = difference
              
                if difference > max_diff[label]:
                    saveImage(inputs, label, '_Max_Diff', model_Type, title)
                    max_diff[label] = difference

                           
            # Add true labels and predicted labels to lists
            true_labels.append(label)
            predicted_labels.append(predicted)
    
    complete_accuracy = (100 * correct / total)
            
    print('\nAccuracy of the network on the %d test audios: %.3f %%\n' %(total,complete_accuracy))
    
    
    # Plot the confusion Matrix
    confusionmatrix(true_labels,predicted_labels,complete_accuracy, model_Type)
	
def saveImage(inputs, label, Type, model_Type, probability_1):
    # get label str from list 
    class_name = INSTRUMENT_FAMILY_CLASSES[label]
    
    # draw the plot
    plt.plot(inputs)
    title = class_name + Type + ':' + str(probability_1)
    plt.title(title)
    
    # save the plot based on net
    if model_Type == 'Best':
        plt.savefig( RESULT_DIR_BEST  + class_name + Type , bbox_inches='tight')
    else:
        plt.savefig( RESULT_DIR_LAST  + class_name + Type , bbox_inches='tight')
    plt.close()	
    

   
def confusionmatrix(true_labels,predicted_labels, accuracy, model_Type):
    """
    This method plots the confusion matrix
    :param cm: Confusion matrix
    :param accuracy: Accuracy of the model
    :param title: Model title
    :return None: None
    """
    
    # Compute the confusion Matrix
    cm = metrics.confusion_matrix(true_labels , predicted_labels)
    
    print('Confusion Matrix')
    print(cm)
    
    plt.figure(figsize=(9,9))
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    title = 'Accuracy : %.3f%%' %accuracy
    plt.title(title, size = 15)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, INSTRUMENT_FAMILY_CLASSES, rotation=45, size = 10)
    plt.yticks(tick_marks, INSTRUMENT_FAMILY_CLASSES, size = 10)
    plt.tight_layout()
    plt.ylabel('Actual', size = 15)
    plt.xlabel('Predicted', size = 15)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
            horizontalalignment='center',
            verticalalignment='center')
    if model_Type == 'Best':
        plt.savefig(RESULT_DIR_BEST + 'ConfusionMatrixModel1', bbox_inches='tight')
    else:
        plt.savefig(RESULT_DIR_LAST + 'ConfusionMatrixModel1', bbox_inches='tight')
    plt.close() 



def create_directories(modelname):
    
    global RESULT_DIR
    global RESULT_DIR_BEST
    global RESULT_DIR_LAST   
    
    RESULT_DIR = modelname + RESULT_DIR
    RESULT_DIR_BEST = modelname + RESULT_DIR_BEST
    RESULT_DIR_LAST = modelname + RESULT_DIR_LAST
      
    print(RESULT_DIR)
        
    
    # Create target Directory if don't exist
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)
        print("Directory " , RESULT_DIR ,  " Created ")
    else:    
        print("Directory " , RESULT_DIR ,  " already exists")
    
    # Create target Directory if don't exist
    if not os.path.exists(RESULT_DIR_BEST):
        os.mkdir(RESULT_DIR_BEST)
        print("Directory " , RESULT_DIR_BEST ,  " Created ")
    else:    
        print("Directory " , RESULT_DIR_BEST ,  " already exists")
        
    # Create target Directory if don't exist
    if not os.path.exists(RESULT_DIR_LAST):
        os.mkdir(RESULT_DIR_LAST)
        print("Directory " , RESULT_DIR_LAST ,  " Created ")
    else:    
        print("Directory " , RESULT_DIR_LAST ,  " already exists")
    
