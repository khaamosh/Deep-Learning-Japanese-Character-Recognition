# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        
        #the architecture of the Linear Network is defined here
        #the value of 784 was created by multiplyting the size of the neural network
        #the value 10 is created since we have 10 features more like 10 kinds of output
        
        self.in_to_hid = torch.nn.Linear(784,10)
        
        #self.hid_to_out = torch.nn.LogSoftmax(2,10)
        
        #the output layer is defined here where we have utilized the LogSoftMax function
        self.hid_out = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        
        #first we reshaped our neural network so that they can be input to the neural network
        re_shape = torch.reshape(x,(x.shape[0],-1))
        
        #the reshaped input is then sent to the hidden layer
        hid_sum = self.in_to_hid(re_shape)

        #here is the activation function 
        out_sum = self.hid_out(hid_sum)

        #the activation function output is then returned here.
        return out_sum # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        
        
        self.in_1_to_hid = torch.nn.Linear(784,150)
        
        self.in_2_to_hid = torch.nn.Linear(150,10)
        
        #activation function
        self.hid_layer= torch.nn.Tanh()
        
        #self.hid_to_out = torch.nn.functional.log_softmax()
        self.hid_to_out = torch.nn.LogSoftmax(dim=1)
        # self.in_1_to_hid_1 = torch.nn.Linear(784,1000)
        
        # #hidden layers at the first hidden layer
        # self.hid_1_to_in_2 = torch.nn.Tanh()
        
        # #input layer 2
        # self.in2_hid_2 = torch.nn.Linear(1000,1000)
        
        # #hidden layer 2
        # self.hid_2_to_in_3 = torch.nn.Tanh()
    
        # self.in_3_to_out = torch.nn.Linear(1000,10)
        
        # #activation functions are defined below
        # self.hid_to_out = torch.nn.LogSoftmax(dim=1)
        

    def forward(self, x):
        #print(x.shape)
        
        re_shape = torch.reshape(x,(x.shape[0],-1))
        
        in_1_out = self.in_1_to_hid(re_shape)
        
        hid_1_out = self.hid_layer(in_1_out)
        
        in_2_out = self.in_2_to_hid(hid_1_out)
        
        #hid_2_out = self.hid_layer(in_2_out)
        
        #output = torch.nn.functional.log_softmax(hid_2_out,dim=1)
        output = self.hid_to_out(in_2_out)
        
        # re_shape = torch.reshape(x,(x.shape[0],-1))
        
        # in_1_out = self.in_1_to_hid_1(re_shape)
        
        # hid_1_out = self.hid_1_to_in_2(in_1_out)
        
        
        # hid_sum_2 = self.in2_hid_2(hid_1_out)
        
        # in_3_to_out_sum = self.hid_2_to_in_3(hid_sum_2)
        
        # in_3_sum = self.in_3_to_out(in_3_to_out_sum)

        # output = self.hid_to_out(in_3_sum)
        
        return output # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        
        self.layer_conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=10,kernel_size=5)
        self.layer_conv_2 = torch.nn.Conv2d(in_channels=10,out_channels=12 , kernel_size=5)
        self.layer_linear = torch.nn.Linear(4800, 100)
        self.layer_output = torch.nn.Linear(100, 10)
        
        #activation function relu function
        self.act = torch.nn.ReLU()
        self.act_log = torch.nn.LogSoftmax(dim=1)

        

    def forward(self, x):
        
        #re_shape = torch.reshape(x, (x.shape[0],-1))
        
        out_input1 = self.layer_conv_1(x)
        
        relu_1 = self.act(out_input1)
        
        out_input2 = self.layer_conv_2(relu_1)
        
        relu_2 = self.act(out_input2)
        
        #print(relu_2.shape)
        
        re_shape = torch.reshape(relu_2, (relu_2.shape[0],-1))
        
        #print(re_shape.shape)
        
        out_linear = self.layer_linear(re_shape)
        
        out_act = self.act(out_linear)
        
        out_line = self.layer_output(out_act)
        
        out_put = self.act_log(out_line)
        
        #output = self.layer_output(out_log)
        
        return out_put # CHANGE CODE HERE
