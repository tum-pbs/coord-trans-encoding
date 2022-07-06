################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Dataset handling
#
################

from torch.utils.data import Dataset
import numpy as np
from os import listdir
import random
from scipy import interpolate
from matplotlib import pyplot as plt
import pickle

# global switch, use fixed max values for dim-less airfoil data?
fixedAirfoilNormalization = False
targetNormalization = True
# global switch, make data dimensionless?
makeDimLess = False
# global switch, remove constant offsets from pressure channel?
removePOffset = False

## helper - compute absolute of inputs or targets
def find_absmax(data, use_targets, x):
    maxval = 0
    for i in range(data.totalLength):
        if use_targets == 0:
            temp_tensor = data.inputs[i]
        else:
            temp_tensor = data.targets[i]
        temp_max = np.max(np.abs(temp_tensor[x]))
        if temp_max > maxval:
            maxval = temp_max
    return maxval

def find_max(data, use_targets, x):
    maxval = -1e12
    for i in range(data.totalLength):
        if use_targets == 0:
            temp_tensor = data.inputs[i]
        else:
            temp_tensor = data.targets[i]
        temp_max = np.max((temp_tensor[x]))
        if temp_max > maxval:
            maxval = temp_max
    return maxval

def find_min(data, use_targets, x):
    minval = 1e12
    for i in range(data.totalLength):
        if use_targets == 0:
            temp_tensor = data.inputs[i]
        else:
            temp_tensor = data.targets[i]
        temp_min = np.min((temp_tensor[x]))
        if temp_min < minval:
            minval = temp_min
    return minval
######################################## DATA LOADER #########################################
#         also normalizes data with max , and optionally makes it dimensionless              #

def LoaderNormalizer(data, isTest = False, shuffle = 0, dataProp = None):
    """
    # data: pass TurbDataset object with initialized dataDir / dataDirTest paths
    # train: when off, process as test data (first load regular for normalization if needed, then replace by test data)
    # dataProp: proportions for loading & mixing 3 different data directories "reg", "shear", "sup"
    #           should be array with [total-length, fraction-regular, fraction-superimposed, fraction-sheared],
    #           passing None means off, then loads from single directory
    """

    if not isTest:
        if dataProp is None:
            # load single directory
            files = listdir(data.dataDir)
            files.sort()
            for i in range(shuffle):
                random.shuffle(files) 
            if isTest:
                print("Reducing data to load for tests")
                files = files[0:min(10, len(files))]
            data.totalLength = len(files)
            data.inputs  = np.empty((len(files), 12, 128, 128))
            # 0- xmach
            # 1- aoa
            # 2- re
            # 3- si4
            # 4- sj1
            # 5- sj3
            # 6- sj4
            # 7- sk1
            # 8- sk3
            # 9- sk4
            data.targets = np.empty((len(files), 4, 128, 128))

            for i, file in enumerate(files):
                npfile = np.load(data.dataDir + file)
                d = npfile['a']
                data.inputs[i, 0:10, :, :] = d[0:10] # xmach to sk4 0,...9
                data.inputs[i, 10, :, :] = d[14] # 14 and 15
                data.inputs[i, 11, :, :] = d[15] # 14 and 15
                data.targets[i] = d[10:14] # 10, 11, 12, 13
            print("Number of data loaded:", len(data.inputs) )
            #print(data.totalLength, data.inputs.shape, data.targets.shape)
            #2500 (2500,3,128,128) (2500,4,128,128) 
        else:
            pass
        ################################## NORMALIZATION OF TRAINING DATA ##########################################



        data.max_inputs_0 = find_max(data, 0, 0) # xmach
        data.max_inputs_1 = find_max(data, 0, 1) # aoa
        data.max_inputs_2 = find_max(data, 0, 2) # re
        data.max_inputs_3 = find_max(data, 0, 3) # si4
        data.max_inputs_4 = find_max(data, 0, 4) # sj1
        data.max_inputs_5 = find_max(data, 0, 5) # sj3
        data.max_inputs_6 = find_max(data, 0, 6) # sj4
        data.max_inputs_7 = find_max(data, 0, 7) # sk1
        data.max_inputs_8 = find_max(data, 0, 8) # sk3
        data.max_inputs_9 = find_max(data, 0, 9) # sk4
        data.max_inputs_10 = 1.25 # x
        data.max_inputs_11 = find_max(data, 0, 11) # y
    
        with open('max_inputs.pickle', 'wb') as f: pickle.dump([data.max_inputs_0,data.max_inputs_1,data.max_inputs_2,data.max_inputs_3,data.max_inputs_4,data.max_inputs_5,data.max_inputs_6,data.max_inputs_7,data.max_inputs_8,data.max_inputs_9,data.max_inputs_10,data.max_inputs_11], f)
        f.close()




        data.min_inputs_0 = find_min(data, 0, 0) # 
        data.min_inputs_1 = find_min(data, 0, 1) # 
        data.min_inputs_2 = find_min(data, 0, 2) # 
        data.min_inputs_3 = find_min(data, 0, 3) # 
        data.min_inputs_4 = find_min(data, 0, 4) # 
        data.min_inputs_5 = find_min(data, 0, 5) # 
        data.min_inputs_6 = find_min(data, 0, 6) # 
        data.min_inputs_7 = find_min(data, 0, 7) # 
        data.min_inputs_8 = find_min(data, 0, 8) # 
        data.min_inputs_9 = find_min(data, 0, 9) # 
        data.min_inputs_10 = find_min(data, 0, 10) # 
        data.min_inputs_11 = find_min(data, 0, 11) # 
        with open('min_inputs.pickle', 'wb') as f: pickle.dump([data.min_inputs_0,data.min_inputs_1,data.min_inputs_2,data.min_inputs_3,data.min_inputs_4,data.min_inputs_5,data.min_inputs_6,data.min_inputs_7,data.min_inputs_8,data.min_inputs_9,data.min_inputs_10,data.min_inputs_11], f)
        f.close()
        if targetNormalization:
            data.max_targets_0 = find_max(data, 1, 0)
            data.max_targets_1 = find_max(data, 1, 1)
            data.max_targets_2 = find_max(data, 1, 2)
            data.max_targets_3 = find_max(data, 1, 3)
            with open('max_targets.pickle', 'wb') as f: pickle.dump([data.max_targets_0,data.max_targets_1,data.max_targets_2,data.max_targets_3], f)
            f.close()

            data.min_targets_0 = find_min(data, 1, 0)
            data.min_targets_1 = find_min(data, 1, 1)
            data.min_targets_2 = find_min(data, 1, 2)
            data.min_targets_3 = find_min(data, 1, 3)
            with open('min_targets.pickle', 'wb') as f: pickle.dump([data.min_targets_0,data.min_targets_1,data.min_targets_2,data.min_targets_3], f)
            f.close()
        else:
            data.max_targets_0 = 1
            data.max_targets_1 = 1
            data.max_targets_2 = 1
            data.max_targets_3 = 1
            with open('max_targets.pickle', 'wb') as f: pickle.dump([data.max_targets_0,data.max_targets_1,data.max_targets_2,data.max_targets_3], f)
            f.close()

            data.min_targets_0 = 0
            data.min_targets_1 = 0
            data.min_targets_2 = 0
            data.min_targets_3 = 0
            with open('min_targets.pickle', 'wb') as f: pickle.dump([data.min_targets_0,data.min_targets_1,data.min_targets_2,data.min_targets_3], f)
            f.close()

#########--below -- to be fixed---
        data.inputs[:,0,:,:] -= data.min_inputs_0
        data.inputs[:,1,:,:] -= data.min_inputs_1
        data.inputs[:,2,:,:] -= data.min_inputs_2# add for xmach, aoa, re
        data.inputs[:,3,:,:] -= data.min_inputs_3
        data.inputs[:,4,:,:] -= data.min_inputs_4
        data.inputs[:,5,:,:] -= data.min_inputs_5
        data.inputs[:,6,:,:] -= data.min_inputs_6
        data.inputs[:,7,:,:] -= data.min_inputs_7
        data.inputs[:,8,:,:] -= data.min_inputs_8
        data.inputs[:,9,:,:] -= data.min_inputs_9
        data.inputs[:,10,:,:] -= data.min_inputs_10
        data.inputs[:,11,:,:] -= data.min_inputs_11

        data.targets[:,0,:,:] -= data.min_targets_0
        data.targets[:,1,:,:] -= data.min_targets_1
        data.targets[:,2,:,:] -= data.min_targets_2
        data.targets[:,3,:,:] -= data.min_targets_3

        data.inputs[:,0,:,:] *= (1.0/(data.max_inputs_0-data.min_inputs_0+1e-20))
        data.inputs[:,1,:,:] *= (1.0/(data.max_inputs_1-data.min_inputs_1+1e-20))
        data.inputs[:,2,:,:] *= (1.0/(data.max_inputs_2-data.min_inputs_2+1e-20)) # add for xmach, aoa, re
        data.inputs[:,3,:,:] *= (1.0/(data.max_inputs_3-data.min_inputs_3))
        data.inputs[:,4,:,:] *= (1.0/(data.max_inputs_4-data.min_inputs_4))
        data.inputs[:,5,:,:] *= (1.0/(data.max_inputs_5-data.min_inputs_5))
        data.inputs[:,6,:,:] *= (1.0/(data.max_inputs_6-data.min_inputs_6))
        data.inputs[:,7,:,:] *= (1.0/(data.max_inputs_7-data.min_inputs_7))
        data.inputs[:,8,:,:] *= (1.0/(data.max_inputs_8-data.min_inputs_8))
        data.inputs[:,9,:,:] *= (1.0/(data.max_inputs_9-data.min_inputs_9))
        data.inputs[:,10,:,:] *= (1.0/(data.max_inputs_10-data.min_inputs_10))
        data.inputs[:,11,:,:] *= (1.0/(data.max_inputs_11-data.min_inputs_11))

        data.targets[:,0,:,:] *= (1.0/(data.max_targets_0-data.min_targets_0))
        data.targets[:,1,:,:] *= (1.0/(data.max_targets_1-data.min_targets_1))
        data.targets[:,2,:,:] *= (1.0/(data.max_targets_2-data.min_targets_2))
        data.targets[:,3,:,:] *= (1.0/(data.max_targets_3-data.min_targets_3))

    ###################################### NORMALIZATION  OF TEST DATA #############################################

    if isTest:
        print("data.dataDirTest:",data.dataDirTest)

        with open('./max_inputs.pickle', 'rb') as f: max_inputs = pickle.load(f)
        f.close()
        with open('./max_targets.pickle', 'rb') as f: max_targets = pickle.load(f)
        f.close()
        print("## max inputs  ##: ",max_inputs) 
        print("## max targets ##: ",max_targets) 
        data.max_inputs_0 = max_inputs[0] # 
        data.max_inputs_1 = max_inputs[1] # 
        data.max_inputs_2 = max_inputs[2] # 
        data.max_inputs_3 = max_inputs[3] # 
        data.max_inputs_4 = max_inputs[4] # 
        data.max_inputs_5 = max_inputs[5] # 
        data.max_inputs_6 = max_inputs[6] # 
        data.max_inputs_7 = max_inputs[7] # 
        data.max_inputs_8 = max_inputs[8] # 
        data.max_inputs_9 = max_inputs[9] # 
        data.max_inputs_10 = max_inputs[10] # 
        data.max_inputs_11 = max_inputs[11] # 

        data.max_targets_0 = max_targets[0]
        data.max_targets_1 = max_targets[1]
        data.max_targets_2 = max_targets[2]
        data.max_targets_3 = max_targets[3]


        with open('./min_inputs.pickle', 'rb') as f: min_inputs = pickle.load(f)
        f.close()
        with open('./min_targets.pickle', 'rb') as f: min_targets = pickle.load(f)
        f.close()
        print("## min inputs  ##: ",min_inputs) 
        print("## min targets ##: ",min_targets) 
        data.min_inputs_0 = min_inputs[0] 
        data.min_inputs_1 = min_inputs[1] 
        data.min_inputs_2 = min_inputs[2] 
        data.min_inputs_3 = min_inputs[3] 
        data.min_inputs_4 = min_inputs[4] 
        data.min_inputs_5 = min_inputs[5] 
        data.min_inputs_6 = min_inputs[6] 
        data.min_inputs_7 = min_inputs[7] 
        data.min_inputs_8 = min_inputs[8] 
        data.min_inputs_9 = min_inputs[9] 
        data.min_inputs_10 = min_inputs[10] 
        data.min_inputs_11 = min_inputs[11] 

        data.min_targets_0 = min_targets[0]
        data.min_targets_1 = min_targets[1]
        data.min_targets_2 = min_targets[2]
        data.min_targets_3 = min_targets[3]





        files = listdir(data.dataDirTest)
        files.sort()
        data.totalLength = len(files)
        data.inputs  = np.empty((len(files), 12, 128, 128))
        data.targets = np.empty((len(files), 4, 128, 128))

        for i, file in enumerate(files):
            npfile = np.load(data.dataDirTest + file)
            d = npfile['a']
            data.inputs[i, 0:10, :, :] = d[0:10] # 0, 1, 2, 
            data.inputs[i, 10, :, :] = d[14] # x
            data.inputs[i, 11, :, :] = d[15] # y
            data.targets[i] = d[10:14] # 10, 11, 12, 13
        print("Number of data loaded:", len(data.inputs) )




#print("Liwei: ", data.max_inputs_0, data.max_inputs_1, data.max_inputs_2) 
        print("Data stats, input  mean %f, max  %f;   targets mean %f , max %f " % ( 
        np.mean(np.abs(data.targets), keepdims=False), np.max(np.abs(data.targets), keepdims=False) , 
        np.mean(np.abs(data.inputs), keepdims=False) , np.max(np.abs(data.inputs), keepdims=False) ) ) 
#########--below -- to be fixed---
        data.inputs[:,0,:,:] -= data.min_inputs_0
        data.inputs[:,1,:,:] -= data.min_inputs_1
        data.inputs[:,2,:,:] -= data.min_inputs_2# add for xmach, aoa, re
        data.inputs[:,3,:,:] -= data.min_inputs_3
        data.inputs[:,4,:,:] -= data.min_inputs_4
        data.inputs[:,5,:,:] -= data.min_inputs_5
        data.inputs[:,6,:,:] -= data.min_inputs_6
        data.inputs[:,7,:,:] -= data.min_inputs_7
        data.inputs[:,8,:,:] -= data.min_inputs_8
        data.inputs[:,9,:,:] -= data.min_inputs_9
        data.inputs[:,10,:,:] -= data.min_inputs_10
        data.inputs[:,11,:,:] -= data.min_inputs_11

        data.targets[:,0,:,:] -= data.min_targets_0
        data.targets[:,1,:,:] -= data.min_targets_1
        data.targets[:,2,:,:] -= data.min_targets_2
        data.targets[:,3,:,:] -= data.min_targets_3

        data.inputs[:,0,:,:] *= (1.0/(data.max_inputs_0-data.min_inputs_0+1e-20))
        data.inputs[:,1,:,:] *= (1.0/(data.max_inputs_1-data.min_inputs_1+1e-20))
        data.inputs[:,2,:,:] *= (1.0/(data.max_inputs_2-data.min_inputs_2+1e-20)) # add for xmach, aoa, re
        data.inputs[:,3,:,:] *= (1.0/(data.max_inputs_3-data.min_inputs_3))
        data.inputs[:,4,:,:] *= (1.0/(data.max_inputs_4-data.min_inputs_4))
        data.inputs[:,5,:,:] *= (1.0/(data.max_inputs_5-data.min_inputs_5))
        data.inputs[:,6,:,:] *= (1.0/(data.max_inputs_6-data.min_inputs_6))
        data.inputs[:,7,:,:] *= (1.0/(data.max_inputs_7-data.min_inputs_7))
        data.inputs[:,8,:,:] *= (1.0/(data.max_inputs_8-data.min_inputs_8))
        data.inputs[:,9,:,:] *= (1.0/(data.max_inputs_9-data.min_inputs_9))
        data.inputs[:,10,:,:] *= (1.0/(data.max_inputs_10-data.min_inputs_10))
        data.inputs[:,11,:,:] *= (1.0/(data.max_inputs_11-data.min_inputs_11))

        data.targets[:,0,:,:] *= (1.0/(data.max_targets_0-data.min_targets_0))
        data.targets[:,1,:,:] *= (1.0/(data.max_targets_1-data.min_targets_1))
        data.targets[:,2,:,:] *= (1.0/(data.max_targets_2-data.min_targets_2))
        data.targets[:,3,:,:] *= (1.0/(data.max_targets_3-data.min_targets_3))





    return data

######################################## DATA SET CLASS #########################################

class TurbDataset(Dataset):

    # mode "enum" , pass to mode param of TurbDataset (note, validation mode is not necessary anymore)
    TRAIN = 0
    TEST  = 2

    def __init__(self, dataProp=None, mode=TRAIN, dataDir="../data/train/", dataDirTest="../data/test/", shuffle=0, normMode=0):
        global makeDimLess, removePOffset
        """
        :param dataProp: for split&mix from multiple dirs, see LoaderNormalizer; None means off
        :param mode: TRAIN|TEST , toggle regular 80/20 split for training & validation data, or load test data
        :param dataDir: directory containing training data
        :param dataDirTest: second directory containing test data , needs training dir for normalization
        :param normMode: toggle normalization
        """
        if not (mode==self.TRAIN or mode==self.TEST):
            print("Error - TurbDataset invalid mode "+format(mode) ); exit(1)

        if normMode==1:	
            print("Warning - poff off!!")
            removePOffset = False
        if normMode==2:	
            print("Warning - poff and dimless off!!!")
            makeDimLess = False
            removePOffset = False

        self.mode = mode
        self.dataDir = dataDir
        self.dataDirTest = dataDirTest # only for mode==self.TEST

        # load & normalize data
        self = LoaderNormalizer(self, isTest=(mode==self.TEST), dataProp=dataProp, shuffle=shuffle)

        if not self.mode==self.TEST:
            # split for train/validation sets (80/20) , max 400
            targetLength = self.totalLength - min( int(self.totalLength*0.2) , 400)

            self.valiInputs = self.inputs[targetLength:]
            self.valiTargets = self.targets[targetLength:]
            self.valiLength = self.totalLength - targetLength

            self.inputs = self.inputs[:targetLength]
            self.targets = self.targets[:targetLength]
            self.totalLength = self.inputs.shape[0]

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    #  reverts normalization 
    def denormalize(self, data, v_norm):
        a = data.copy()
        a[0,:,:] /= (1.0/(self.max_targets_0-self.min_targets_0))
        a[1,:,:] /= (1.0/(self.max_targets_1-self.min_targets_1))
        a[2,:,:] /= (1.0/(self.max_targets_2-self.min_targets_2))
        a[3,:,:] /= (1.0/(self.max_targets_3-self.min_targets_3))
        a[0,:,:] += self.min_targets_0
        a[1,:,:] += self.min_targets_1
        a[2,:,:] += self.min_targets_2
        a[3,:,:] += self.min_targets_3


        print("Liwei: makeDimLess in denormalize routine max_targets=", self.max_targets_0, self.max_targets_1, self.max_targets_2, self.max_targets_3)
        print("Liwei: makeDimLess in denormalize routine min_targets=", self.min_targets_0, self.min_targets_1, self.min_targets_2, self.min_targets_3)
        return a

# simplified validation data set (main one is TurbDataset above)

class ValiDataset(TurbDataset):
    def __init__(self, dataset): 
        self.inputs = dataset.valiInputs
        self.targets = dataset.valiTargets
        self.totalLength = dataset.valiLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

