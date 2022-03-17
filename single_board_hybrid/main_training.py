import torch
import torch.nn as nn
import time
import argparse
import os
import datetime

from torch.distributions.categorical import Categorical

# visualization
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import math
import numpy as np
import torch.nn.functional as F
import random
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
####### my own import file ##########
from listofpathpoint import input_handler
import cnc_input
from hybrid_models import HPN
####### my own import file ##########

'''
above part I designed the rectangle-characterized TSP, that means for every step the agent walk through a corner,
then he travel through the whole rectangle using zig-zag, finally he ends up at one of the rest corners of 
the rextangle, so, it equals the agent walk through three points at one step, in practice, I add three points into 
mask to make them unselectable.
'''
def rectangle_process(idx,Y,mask):
    Y1 = Y[zero_to_bsz, idx.data].clone()
    Y0 = None
    # ***************# my own design for the situation of the rectangle field
    # my arrangement of one rectangle is four corners: left-up ,right-up, right-down, left-down
    if (Y1[1] == Y[zero_to_bsz, idx.data + 1][1]):
        if (Y[0] == Y[zero_to_bsz, idx.data - 1][0]):
            # this is a right-down point
            kind_temp = input_handler()
            kind = kind_temp.is_odd_is_row(Y[zero_to_bsz, idx.data - 2], Y[zero_to_bsz, idx.data - 1],\
                                           Y1, Y[zero_to_bsz, idx.data + 1])
            if kind == (0, 0):  # ends up at right-up point
                if k == 0:
                    Y_ini = Y[zero_to_bsz, idx.data - 1]
                if k > 0:
                    reward = torch.sum((Y1 - Y0) ** 2, dim=1) ** 0.5
                    reward += torch.sum((Y[zero_to_bsz, idx.data - 1] - Y1) ** 2, dim=1) ** 0.5
                Y0 = Y[zero_to_bsz, idx.data - 1].clone()
                x = Y[zero_to_bsz, idx.data - 1].clone()
                R += reward
                logprobs += torch.log(output[zero_to_bsz, idx.data] + TINY)
                mask[zero_to_bsz, idx.data] += -np.inf
                mask[zero_to_bsz, idx.data - 2] += -np.inf
                mask[zero_to_bsz, idx.data + 1] += -np.inf
            elif kind == (0, 1):  # ends up at left-down point
                if k == 0:
                    Y_ini = Y[zero_to_bsz, idx.data + 1]
                if k > 0:
                    reward = torch.sum((Y1 - Y0) ** 2, dim=1) ** 0.5
                    reward += torch.sum((Y[zero_to_bsz, idx.data + 1] - Y1) ** 2, dim=1) ** 0.5
                Y0 = Y[zero_to_bsz, idx.data + 1].clone()
                x = Y[zero_to_bsz, idx.data + 1].clone()
                R += reward
                logprobs += torch.log(output[zero_to_bsz, idx.data] + TINY)
                mask[zero_to_bsz, idx.data] += -np.inf
                mask[zero_to_bsz, idx.data - 2] += -np.inf
                mask[zero_to_bsz, idx.data - 1] += -np.inf
            else:  # ends up at left-up point
                if k == 0:
                    Y_ini = Y[zero_to_bsz, idx.data - 2]
                if k > 0:
                    reward = torch.sum((Y1 - Y0) ** 2, dim=1) ** 0.5
                    reward += torch.sum((Y[zero_to_bsz, idx.data - 2] - Y1) ** 2, dim=1) ** 0.5
                Y0 = Y[zero_to_bsz, idx.data - 2].clone()
                x = Y[zero_to_bsz, idx.data - 2].clone()
                R += reward
                logprobs += torch.log(output[zero_to_bsz, idx.data] + TINY)
                mask[zero_to_bsz, idx.data] += -np.inf
                mask[zero_to_bsz, idx.data + 1] += -np.inf
                mask[zero_to_bsz, idx.data - 1] += -np.inf
        else:
            # this is a left-up point
            kind_temp = input_handler()
            kind = kind_temp.is_odd_is_row(Y1, Y[zero_to_bsz, idx.data + 1], Y[zero_to_bsz, idx.data + 2],\
                                           Y[zero_to_bsz, idx.data + 3])
            if kind == (0, 0):  # ends up at left-down point
                if k == 0:
                    Y_ini = Y[zero_to_bsz, idx.data + 3]
                if k > 0:
                    reward = torch.sum((Y1 - Y0) ** 2, dim=1) ** 0.5
                    reward += torch.sum((Y[zero_to_bsz, idx.data + 3] - Y1) ** 2, dim=1) ** 0.5
                Y0 = Y[zero_to_bsz, idx.data + 3].clone()
                x = Y[zero_to_bsz, idx.data + 3].clone()
                R += reward
                logprobs += torch.log(output[zero_to_bsz, idx.data] + TINY)
                mask[zero_to_bsz, idx.data] += -np.inf
                mask[zero_to_bsz, idx.data + 2] += -np.inf
                mask[zero_to_bsz, idx.data + 1] += -np.inf
            elif kind == (0, 1):  # ends up at right-up point
                if k == 0:
                    Y_ini = Y[zero_to_bsz, idx.data + 2]
                if k > 0:
                    reward = torch.sum((Y1 - Y0) ** 2, dim=1) ** 0.5
                    reward += torch.sum((Y[zero_to_bsz, idx.data + 2] - Y1) ** 2, dim=1) ** 0.5
                Y0 = Y[zero_to_bsz, idx.data + 2].clone()
                x = Y[zero_to_bsz, idx.data + 2].clone()
                R += reward
                logprobs += torch.log(output[zero_to_bsz, idx.data] + TINY)
                mask[zero_to_bsz, idx.data] += -np.inf
                mask[zero_to_bsz, idx.data + 3] += -np.inf
                mask[zero_to_bsz, idx.data + 1] += -np.inf
            else:  # ends up at right-down point
                if k == 0:
                    Y_ini = Y[zero_to_bsz, idx.data + 1]
                if k > 0:
                    reward = torch.sum((Y1 - Y0) ** 2, dim=1) ** 0.5
                    reward += torch.sum((Y[zero_to_bsz, idx.data + 1] - Y1) ** 2, dim=1) ** 0.5
                Y0 = Y[zero_to_bsz, idx.data + 1].clone()
                x = Y[zero_to_bsz, idx.data + 1].clone()
                R += reward
                logprobs += torch.log(output[zero_to_bsz, idx.data] + TINY)
                mask[zero_to_bsz, idx.data] += -np.inf
                mask[zero_to_bsz, idx.data + 2] += -np.inf
                mask[zero_to_bsz, idx.data + 3] += -np.inf
    else:
        if (Y[0] == Y[zero_to_bsz, idx.data + 1][0]):
            # this is a right-up point
            kind_temp = input_handler()
            kind = kind_temp.is_odd_is_row(Y[zero_to_bsz, idx.data - 1], Y1, Y[zero_to_bsz, idx.data + 1],\
                                           Y[zero_to_bsz, idx.data + 2])
            if kind == (0, 0):  # ends up at right-down point
                if k == 0:
                    Y_ini = Y[zero_to_bsz, idx.data + 1]
                if k > 0:
                    reward = torch.sum((Y1 - Y0) ** 2, dim=1) ** 0.5
                    reward += torch.sum((Y[zero_to_bsz, idx.data + 1] - Y1) ** 2, dim=1) ** 0.5
                Y0 = Y[zero_to_bsz, idx.data + 1].clone()
                x = Y[zero_to_bsz, idx.data + 1].clone()
                R += reward
                logprobs += torch.log(output[zero_to_bsz, idx.data] + TINY)
                mask[zero_to_bsz, idx.data] += -np.inf
                mask[zero_to_bsz, idx.data + 2] += -np.inf
                mask[zero_to_bsz, idx.data - 1] += -np.inf
            elif kind == (0, 1):  # ends up at left-up point
                if k == 0:
                    Y_ini = Y[zero_to_bsz, idx.data - 1]
                if k > 0:
                    reward = torch.sum((Y1 - Y0) ** 2, dim=1) ** 0.5
                    reward += torch.sum((Y[zero_to_bsz, idx.data - 1] - Y1) ** 2, dim=1) ** 0.5
                Y0 = Y[zero_to_bsz, idx.data - 1].clone()
                x = Y[zero_to_bsz, idx.data - 1].clone()
                R += reward
                logprobs += torch.log(output[zero_to_bsz, idx.data] + TINY)
                mask[zero_to_bsz, idx.data] += -np.inf
                mask[zero_to_bsz, idx.data + 1] += -np.inf
                mask[zero_to_bsz, idx.data + 2] += -np.inf
            else:  # ends up at left-down point
                if k == 0:
                    Y_ini = Y[zero_to_bsz, idx.data + 2]
                if k > 0:
                    reward = torch.sum((Y1 - Y0) ** 2, dim=1) ** 0.5
                    reward += torch.sum((Y[zero_to_bsz, idx.data + 2] - Y1) ** 2, dim=1) ** 0.5
                Y0 = Y[zero_to_bsz, idx.data + 2].clone()
                x = Y[zero_to_bsz, idx.data + 2].clone()
                R += reward
                logprobs += torch.log(output[zero_to_bsz, idx.data] + TINY)
                mask[zero_to_bsz, idx.data] += -np.inf
                mask[zero_to_bsz, idx.data + 2] += -np.inf
                mask[zero_to_bsz, idx.data - 1] += -np.inf
        else:
            # this is a left-down point
            kind_temp = input_handler()
            kind = kind_temp.is_odd_is_row(Y[zero_to_bsz, idx.data - 3], Y[zero_to_bsz, idx.data - 2], \
                                           Y[zero_to_bsz, idx.data - 1], Y1)
            if kind == (0, 0):  # ends up at left-up point
                if k == 0:
                    Y_ini = Y[zero_to_bsz, idx.data - 3]
                if k > 0:
                    reward = torch.sum((Y1 - Y0) ** 2, dim=1) ** 0.5
                    reward += torch.sum((Y[zero_to_bsz, idx.data - 3] - Y1) ** 2, dim=1) ** 0.5
                Y0 = Y[zero_to_bsz, idx.data - 3].clone()
                x = Y[zero_to_bsz, idx.data - 3].clone()
                R += reward
                logprobs += torch.log(output[zero_to_bsz, idx.data] + TINY)
                mask[zero_to_bsz, idx.data] += -np.inf
                mask[zero_to_bsz, idx.data - 2] += -np.inf
                mask[zero_to_bsz, idx.data - 1] += -np.inf
            elif kind == (0, 1):  # ends up at right-down point
                if k == 0:
                    Y_ini = Y[zero_to_bsz, idx.data - 1]
                if k > 0:
                    reward = torch.sum((Y1 - Y0) ** 2, dim=1) ** 0.5
                    reward += torch.sum((Y[zero_to_bsz, idx.data - 1] - Y1) ** 2, dim=1) ** 0.5
                Y0 = Y[zero_to_bsz, idx.data - 1].clone()
                x = Y[zero_to_bsz, idx.data - 1].clone()
                R += reward
                logprobs += torch.log(output[zero_to_bsz, idx.data] + TINY)
                mask[zero_to_bsz, idx.data] += -np.inf
                mask[zero_to_bsz, idx.data - 3] += -np.inf
                mask[zero_to_bsz, idx.data - 2] += -np.inf
            else:  # ends up at right-up point
                if k == 0:
                    Y_ini = Y[zero_to_bsz, idx.data - 2]
                if k > 0:
                    reward = torch.sum((Y1 - Y0) ** 2, dim=1) ** 0.5
                    reward += torch.sum((Y[zero_to_bsz, idx.data - 2] - Y1) ** 2, dim=1) ** 0.5
                Y0 = Y[zero_to_bsz, idx.data - 2].clone()
                x = Y[zero_to_bsz, idx.data - 2].clone()
                R += reward
                logprobs += torch.log(output[zero_to_bsz, idx.data] + TINY)
                mask[zero_to_bsz, idx.data] += -np.inf
                mask[zero_to_bsz, idx.data - 3] += -np.inf
                mask[zero_to_bsz, idx.data - 1] += -np.inf
    return mask, R, logprobs,Y0,Y1,Y_ini,x
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cpu");
gpu_id = -1  # select CPU

gpu_id = '0'  # select a single GPU
# gpu_id = '2,3' # select multiple GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU name: {:s}, gpu_id: {:s}'.format(torch.cuda.get_device_name(0), gpu_id))

print(device)
'''
so, the models we have are TransEncoderNet,
                            Attention
                            LSTM
                            HPN
each one have initial parameters and the forward part, 
once we have the forward part, the back propagation will 
finished automatically by pytorch  
'''
TOL = 1e-3
TINY = 1e-15
learning_rate = 1e-4   #learning rate
B = 512             #batch size
B_val = 1000        #validation Batchsize
B_valLoop = 20
steps = 2500
n_epoch = 100       # epochs

print('======================')
print('prepare to train')
print('======================')
print('Hyper parameters:')
print('learning rate', learning_rate)
print('batch size', B)
print('validation size', B_val)
print('steps', steps)
print('epoch', n_epoch)
print('======================')

'''
instantiate a training network and a baseline network
'''
temp = input_handler('mother_board.json')
X_val = temp.every_point()
X_val = torch.FloatTensor(X_val)
print(len(X_val))
'''
X_val consisted by 'list of list of list'
'rectangle list' 'channel list' 'point xy list' respectively
'''
try:
    del Actor  # remove existing model
    del Critic # remove existing model
except:
    pass
Actor = HPN(n_feature = 2, n_hidden = 128)
Critic = HPN(n_feature = 2, n_hidden = 128)
optimizer = optim.Adam(Actor.parameters(), lr=learning_rate)

# Putting Critic model on the eval mode
Actor = Actor.to(device)
Critic = Critic.to(device)
Critic.eval()

epoch_ckpt = 0
tot_time_ckpt = 0

val_mean = []
val_std = []

plot_performance_train = []
plot_performence_baseline = []
# recording the result of the resent epoch makes it available for future
#*********************# Uncomment these lines to load the previous check point
"""
checkpoint_file = "filename of the .pkl"
checkpoint = torch.load(checkpoint_file, map_location=device)
epoch_ckpt = checkpoint['epoch'] + 1
tot_time_ckpt = checkpoint['tot_time']
plot_performance_train = checkpoint['plot_performance_train']
plot_performance_baseline = checkpoint['plot_performance_baseline']
Critic.load_state_dict(checkpoint['model_baseline'])
Actor.load_state_dict(checkpoint['model_train'])
optimizer.load_state_dict(checkpoint['optimizer'])

print('Re-start training with saved checkpoint file={:s}\n  Checkpoint at epoch= {:d} and time={:.3f}min\n'.format(checkpoint_file,epoch_ckpt-1,tot_time_ckpt/60))

"""
#***********************# Uncomment these lines to load the previous check point

# Main training loop
# The core training concept mainly upon Sampling from the actor
# then taking the greedy action from the critic


start_training_time = time.time()
time_stamp = datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S") # Load the time stamp

C = 0       # baseline => the object which the actor can compare
R = 0       # reward

zero_to_bsz = torch.arange(B, device = device) # a list contains 0 to (batch size -1)

for epoch in range(0, n_epoch):
    # re-start training with saved checkpoint
    epoch += epoch_ckpt # adding the number of the former epochs

    # Train the model for one epoch

    start = time.time() # record the starting time
    Actor.train() #start training actor

    for i in range(1, steps+1): # 1 ~ 2500 steps
        X = X_val
        mask = torch.zeros(B,len(X)).cuda() # use mask to make some points impossible to choose
        R= 0
        logprobs = 0
        reward = 0
        Y = X.view(B,len(X),2)
        x = Y[:,0] #set the single batch to the x 
        h = None
        c = None
        context = None
        Transcontext = None

        # Actor Sampling phase
        for k in range(len(X)):
            context, Transcontext, output, h, c, _ = Actor(context,Transcontext,x=x, X_all=X, h=h, c=c, mask=mask)
            sampler = torch.distributions.Categorical(output)
            idx = sampler.sample()
            # prepare for the back propagation of pytorch
            mask, R, logprobs,Y0,Y1,Y_ini,x = rectangle_process(idx, Y, mask)

        R += torch.sum((Y1 - Y_ini)**2,dim=1)**0.5
# critic baseline phase, use the baseline to compute the actual reward of agent at that time
        mask = torch.zero(B,len(X)).cuda() # use mask to make some points impossible to choose
        C = 0
        baseline = 0
        Y = X.view(B,len(X),2)
        x = Y[:,0,:]
        h = None
        c = None
        context = None
        Transcontext = None
        # compute tours for baseline without grad "Cause we want to fix the weights for the critic"
        with torch.no_grad():
            for k in range(len(X)):
                #same as the above part, with the R => C
                context,Transcontext,output, h, c, _ = Critic(context,Transcontext,x=x, X_all=X, h=h, c=c, mask=mask)#X = X_all
                idx = torch.argmax(output, dim=1) # ----> greedy baseline critic
                mask, C, logprobs, Y0, Y1, Y_ini, x = rectangle_process(idx, Y, mask)
        C += torch.sum((Y1 - Y_ini)**2 , dim=1 )**0.5
        ###################
        # Loss and backprop handling 
        ###################
        
        loss = torch.mean((R - C) * logprobs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print('epoch:',epoch,'batch:' ,i,'/',steps,'reward:',R.mean().item())
    time_one_epoch = time.time() - start #recording the work time of one epoch
    time_tot = time.time() - start_training_time + tot_time_ckpt
    ###################
    # Evaluate train model and baseline 
    # in this phase we just solve random instances with the actor and the critic
    # compare this soluation if we get any improvment we'll transfer the actor's
    # weights into the critic
    ###################
    Actor.eval()
    
    mean_tour_length_actor = 0
    mean_tour_length_critic = 0

    for step in range(0,B_valLoop):
        
        # compute tour for model and baseline
        X = X_temp.sample_rectangele(len(X))   
        X = torch.Tensor(X).cuda()

        mask = torch.zeros(B,len(X)).cuda()
        R = 0
        reward = 0
        Y = X.view(B,len(X),2)
        x = Y[:,0,:]
        
        h = None
        c = None
        context = None
        Transcontext = None

        with torch.no_grad():
            for k in range(len(X)):
                #same as the above part
                context,Transcontext,output, h, c, _ = Actor(context,Transcontext,x=x, X_all=X, h=h, c=c, mask=mask)
                idx = torch.argmax(output, dim=1)
                mask, R, logprobs, Y0, Y1, Y_ini, x = rectangle_process(idx, Y, mask)
        R += torch.sum((Y1 - Y_ini)**2 , dim=1 )**0.5
        # critic baseline
        mask = torch.zeros(B,size).cuda()
        C = 0
        baseline = 0
        
        Y = X.view(B,len(X),2)
        x = Y[:,0,:]
        
        h = None
        c = None
        context = None
        Transcontext = None
        
        with torch.no_grad():
            for k in range(len(X)):
                #same as the above part
                context,Transcontext,output, h, c, _ = Actor(context,Transcontext,x=x, X_all=X, h=h, c=c, mask=mask)
                idx = torch.argmax(output, dim=1)
                mask, C, logprobs, Y0, Y1, Y_ini, x = rectangle_process(idx, Y, mask)
        C  += torch.sum((Y1 - Y_ini)**2 , dim=1 )**0.5
        mean_tour_length_actor  += R.mean().item()
        mean_tour_length_critic += C.mean().item()

    mean_tour_length_actor  =  mean_tour_length_actor  / B_valLoop
    mean_tour_length_critic =  mean_tour_length_critic / B_valLoop
    # evaluate train model and baseline and update if train model is better

    update_baseline = mean_tour_length_actor + TOL < mean_tour_length_critic

    print('Avg Actor {} --- Avg Critic {}'.format(mean_tour_length_actor,mean_tour_length_critic))

    if update_baseline:
        Critic.load_state_dict(Actor.state_dict())
        print('My actor is going on the right road Hallelujah :) Updated')
    ###################
    # Valdiation train model and baseline on 1k random TSP instances
    ###################
    with torch.no_grad():
        # greedy validation
        tour_len = 0
        X = X_val
        mask = torch.zeros(B_val,len(X)).cuda() # this len(X) represent the validation size of TSP
        R = 0
        reward = 0

        Y = X.view(B_val, len(X), 2)    # to the same batch size## this len(X) represent the validation size of TSP
        x = Y[:,0,:]
        h = None
        c = None
        context = None
        Transcontext = None
        for k in range(len(X)):
            #same as the above part 
            context, Transcontext, output, h, c, _ = Actor(context, Transcontext, x=x, X_all=X, h=h, c=c, mask=mask)
            idx = torch.argmax(output, dim=1)
            mask, R, logprobs, Y0, Y1, Y_ini, x = rectangle_process(idx, Y, mask)

        R  += torch.sum((Y1 - Y_ini)**2 , dim=1 )**0.5
        tour_len += R.mean().item()

        print('validation tour length:', tour_len)

    # For checkpoint
    plot_performance_train.append([(epoch+1), mean_tour_length_actor])
    plot_performance_baseline.append([(epoch+1), mean_tour_length_critic])
    # compute the optimally gap ==> this is interesting because there is no LKH or other optimal algorithms 
    # for the problem like this rectangle characterized map
    mystring_min = 'Epoch: {:d}, epoch time: {:.3f}min, tot time: {:.3f}day, L_actor: {:.3f}, L_critic: {:.3f}, gap_train(%): {:.3f}, update: {}'.format(
        epoch, time_one_epoch/60, time_tot/86400, mean_tour_length_actor, mean_tour_length_critic, 100 * gap_train, update_baseline)

    print(mystring_min)
    print('Save Checkpoints')

    # Saving checkpoint
    checkpoint_dir = os.path.join("checkpoint")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    torch.save({
        'epoch': epoch,
        'time': time_one_epoch,
        'tot_time': time_tot,
        'loss': loss.item(),
        'plot_performance_train': plot_performance_train,
        'plot_performance_baseline': plot_performance_baseline,
        'mean_tour_length_val': tour_len,
        'model_baseline': Critic.state_dict(),
        'model_train': Actor.state_dict(),
        'optimizer': optimizer.state_dict(),
        },'{}.pkl'.format(checkpoint_dir + "/checkpoint_" + time_stamp + "-n{}".format(size) + "-gpu{}".format(gpu_id)))



  




                
            
                
        
                        