# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:52:39 2018

@author: Parinitha Nagaraja and Qiaoran Li 
"""
import matplotlib
matplotlib.use('pdf')    
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import modelhelper

#Using CUDA if we have a GPU that supports it along with the correct
#Install, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running model 1 on', device)

RESULT_DIR = 'model1_result/'
RESULT_DIR_BEST = 'model1_result/model1_result_best/'
RESULT_DIR_LAST = 'model1_result/model1_result_last/'

# Network Parameters
INPUT_SIZE = 160
HIDDEN_SIZE = 128
NUM_LAYERS = 3

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers= NUM_LAYERS,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(HIDDEN_SIZE, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(device)
        c0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(device)
        
        r_out, (h_n, h_c) = self.rnn(x, (h0,c0))   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        #out = F.softmax(out)
        return out
    
    

def main():
    
    RUN_ON_SERVER = True
    
    if str(device) == 'cpu': 
        RUN_ON_SERVER = False
    
    want_to_test = input('Do you want to test the model? (1-yes, 0-No): ')     
        
    modelhelper.create_directories('model1')
    
    trainloader, validationloader, test_loader =  modelhelper.divide_datasets(RUN_ON_SERVER, want_to_test)
    
    net = RNN()
    
    if want_to_test == '1':
        modelfilepath = input("Enter the path of model file: ")
        if RUN_ON_SERVER:
            net.load_state_dict(torch.load(modelfilepath))
        else:
            net.load_state_dict(torch.load(modelfilepath, map_location='cpu'))
        modelhelper.compute_accuracy(net,test_loader, RUN_ON_SERVER, 'Last')
    else:
        bestnet, net = modelhelper.training(net, trainloader, validationloader, RUN_ON_SERVER)    
        # compute accuracy using the best model
        print('============ Using best model before overfit ============')
        modelhelper.compute_accuracy(bestnet, test_loader, RUN_ON_SERVER, 'Best')
    
        # compute accuracy using the last model
        print('================== Using last model =====================')
        modelhelper.compute_accuracy(net,test_loader, RUN_ON_SERVER, 'Last')
        
  

if __name__ == '__main__':
    main()
