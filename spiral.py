# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math 

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        #the architecture is defined here for the neural network
        self.input_layer = torch.nn.Linear(2,num_hid)
        self.input_layer_1 = torch.nn.Linear(num_hid,1)
        
        #activation functions
        self.hidden = torch.nn.Tanh()
        self.output = torch.nn.Sigmoid()
        
        self.out_act = 0

    def forward(self, input):
        
        #changing into polar co-ordinates
        
        x = input[:,0]
        y = input[:,1]
        
        r=torch.sqrt(x*x + y*y)
        
        new_r = torch.reshape(r, ( r.shape[0],-1) )
        
        #print(new_r.shape)
        
        a=torch.atan2(y,x)
        new_a = torch.reshape(a, ( a.shape[0],-1) )
        
        #print(new_a.shape)
        
        #input_vals = torch.cartesian_prod(r,a)
        
        input_vals = torch.cat((new_r,new_a),1)
        
        #print(input_vals.shape)
        
        out_in_layer = self.input_layer(input_vals)
        
        self.out_act = self.hidden(out_in_layer)
        
        out_in_1 = self.input_layer_1(self.out_act)
        
        output = self.output(out_in_1)
        
        #output = 0*input[:,0] # CHANGE CODE HERE
        #print("Here")
        
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        
        #network architecture
        self.input_layer = torch.nn.Linear(2,num_hid)
        self.input_layer_1 = torch.nn.Linear(num_hid,num_hid)
        self.output_layer = torch.nn.Linear(num_hid, 1)
        
        #activation functions
        self.act = torch.nn.Tanh()
        self.sig = torch.nn.Sigmoid()
        
        
        self.out_act = 0
        self.out_act_1 = 0
        

    def forward(self, input):
        
        #x = input[:,0]
        #y = input[:,1]
        
        #print(x.shape)
        #print(y.shape)
        
        #new_x = torch.reshape(x,(x.shape[0],-1))
        
        #new_y = torch.reshape(y,(y.shape[0],-1))
        
        #input_vals = torch.cat((new_x,new_y),1)
        
        #print(input_vals.shape)
        
        out_in = self.input_layer(input)
        
        self.out_act = self.act(out_in)
        
        out_in_1  =self.input_layer_1(self.out_act)
        
        self.out_act_1 = self.act(out_in_1)
        
        output_out = self.output_layer(self.out_act_1)
        
        output = self.sig(output_out)
        
        #output = 0*input[:,0] # CHANGE CODE HERE
        
        return output

def graph_hidden(net, layer, node):
    #plt.clf()
    # INSERT CODE HERE
    
    #Refrencing the hidden function given in the assignment and changed the plotting function accordingly
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)
    
    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        output = net(grid)
        
        
        if layer ==1 :
            pred = (net.out_act[:,node]>=0).float()
        
        if layer == 2 :
            pred = (net.out_act_1[:,node]>=0).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
