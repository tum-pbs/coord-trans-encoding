################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Main training script
#
################

import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from DfpNet import TurbNetG, weights_init
import dataset
import utils

######## Settings ########

# number of training iterations
iterations = 400000
# batch size
batch_size = 5
# learning rate, generator
lrG = 0.0006
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 7
# data set config
prop=None # by default, use all from "../data/train"
# save txt files with per epoch loss?
saveL1 = True
saveModel = True
n_save_model = 100
##########################

prefix = ""
if len(sys.argv)>1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

dropout    = 0.      # note, the original runs from https://arxiv.org/abs/1810.08217 used slight dropout, but the effect is minimal; conv layers "shouldn't need" dropout, hence set to 0 here.
doLoad     = ""      # optional, path to pre-trained model

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))
print("Dropout: {}".format(dropout))

##########################

seed = random.randint(0, 2**32 - 1)
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#torch.backends.cudnn.deterministic=True # warning, slower

# create pytorch data object with dfp dataset
data = dataset.TurbDataset(prop, dataDir="/home/liwei/Simulations/cfl3d_TestCases/Z08_transonic/BASIC_data_coordinates_final_metricsAll_1940/train_avg/", dataDirTest="../../BASIC_data_coordinates_final_metricsAll_1940/test_avg/", shuffle=1)

trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
print("Training batches: {}".format(len(trainLoader)))
dataValidation = dataset.ValiDataset(data)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=True, drop_last=True) 
print("Validation batches: {}".format(len(valiLoader)))

# setup training
epochs = int(iterations/len(trainLoader) + 0.5)
netG = TurbNetG(channelExponent=expo, dropout=dropout)
print(netG) # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized TurbNet with {} trainable params ".format(params))

netG.apply(weights_init)
if len(doLoad)>0:
    netG.load_state_dict(torch.load(doLoad))
    print("Loaded model "+doLoad)
netG.cuda()

criterionL1 = nn.L1Loss()
criterionL1.cuda()

optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

targets = Variable(torch.FloatTensor(batch_size, 4, 128, 128))
inputs  = Variable(torch.FloatTensor(batch_size, 12, 128, 128))
targets = targets.cuda()
inputs  = inputs.cuda()

##########################

for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch+1),epochs))

    netG.train()
    L1_accum = 0.0
    samples_accum = 0
    for i, traindata in enumerate(trainLoader, 0):
        inputs_cpu, targets_cpu = traindata
        current_batch_size = targets_cpu.size(0)

        inputs_cpu = inputs_cpu.float().cuda()
        targets_cpu = targets_cpu.float().cuda()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        # compute LR decay
        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG*0.1, lrG)
            if currLr < lrG:
                for g in optimizerG.param_groups:
                    g['lr'] = currLr

        netG.zero_grad()
        gen_out = netG(inputs)

############################################################# ----  work here!

        lossL1 = criterionL1(gen_out, targets)
        lossL1.backward()

        optimizerG.step()

        lossL1viz = lossL1.item()
        L1_accum += lossL1viz
        samples_accum += current_batch_size

        if i==len(trainLoader)-1:
            logline = "Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch, i, lossL1viz)
            print(logline)


    # validation
    netG.eval()
    L1val_accum = 0.0
    for i, validata in enumerate(valiLoader, 0):
        inputs_cpu, targets_cpu = validata
        current_batch_size = targets_cpu.size(0)

        targets_cpu = targets_cpu.float().cuda()
        inputs_cpu = inputs_cpu.float().cuda()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        outputs = netG(inputs)
############################################################# ----  work here!

        lossL1 = criterionL1(outputs, targets)
        L1val_accum += lossL1.item()


    # data for graph plotting
    L1_accum    /= len(trainLoader)
    L1val_accum /= len(valiLoader)
    if saveL1:
        if epoch==0: 
            utils.resetLog(prefix + "L1.txt"   )
            utils.resetLog(prefix + "L1val.txt")
        utils.log(prefix + "L1.txt"   , "{} ".format(L1_accum), False)
        utils.log(prefix + "L1val.txt", "{} ".format(L1val_accum), False)
    
    if saveModel:
        if epoch % n_save_model == 0:
            print("++++++++++ Save model... modelG... ++++++++++")
            torch.save(netG.state_dict(), prefix + "modelG" )

torch.save(netG.state_dict(), prefix + "modelG" )

