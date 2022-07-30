################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Generate training data via OpenFOAM
#
################

import os, math, uuid, sys, random
import numpy as np
import math
#import utils 
import shutil
import subprocess
from matplotlib import pyplot as plt
#samples           = 2500     # no. of datasets to produce
#freestream_angle  = 22.5 #degree  # -angle ... angle

current_dir       = os.getcwd() 
#"/home/liwei/Simulations/cfl3d_TestCases/02_Airfoils/c_type_input_channels/"
seed = random.randint(0, 2**32 - 1)
np.random.seed(seed)
print("Seed: {}".format(seed))
train_metric_dir = "../metric_generation//metric_vol_cfl3d/"
train_mesh = "./train_mesh/"
output_dir ="../BASIC_data_coordinates_final_metricsAll/train_avg/"
raw_dir = "./train_avg/"



################################################################################
#
# Function to read grid
#
################################################################################
def read_grid(fname):

# Open grid file
  f = open(fname)

# Read imax, jmax
# 3D grid specifies number of blocks on top line
  line1 = f.readline()
  flag = len(line1.split())
  if flag == 1:
    threed = True
  else:
    threed = False

  if threed:
    line1 = f.readline()
    imax, kmax, jmax = [int(x) for x in line1.split()]
  else:
    imax, jmax = [int(x) for x in line1.split()]
    kmax = 1

# Read geometry data
  x = np.zeros((imax,jmax))
  y = np.zeros((imax,jmax))
  if threed:
    for j in range(0, jmax):
      for k in range(0, kmax):
        for i in range(0, imax):
          x[i,j] = float(f.readline())
    for j in range(0, jmax):
      for k in range(0, kmax):
        for i in range(0, imax):
          dummy = float(f.readline())
    for j in range(0, jmax):
      for k in range(0, kmax):
        for i in range(0, imax):
          y[i,j] = float(f.readline())
  else:
    for j in range(0, jmax):
      for i in range(0, imax):
        x[i,j] = float(f.readline())

    for j in range(0, jmax):
      for i in range(0, imax):
        y[i,j] = float(f.readline())

# Print message
  print('Successfully read grid file ' + fname)

# Close the file
  f.close

  return (imax, jmax, kmax, x, y, threed)
################################################################################
#
# Function to read Plot3D function file
#
################################################################################
def read_function_file(fname, imax, jmax, kmax, threed):

# Open stats file
  f = open(fname)

# Read first line to get variables category
  line1 = f.readline()
  varcat = line1[1:].rstrip()

# Second line gives variable names
  line1 = f.readline()
  varnames = line1[1:].rstrip()
  variables = varnames.split(", ")

# Number of variables
  nvars = len(variables)

# Initialize data and skip the next line
  values = np.zeros((nvars,imax,jmax))
  maxes = np.zeros((nvars))*-1000.0
  mins = np.ones((nvars))*1000.0
  line1 = f.readline()

# Read grid stats data, storing min and max
  for n in range(0, nvars):
    if (threed):
      for j in range(0, jmax):
        for k in range(0, kmax):
          for i in range(0, imax):
            values[n,i,j] = float(f.readline())
            if values[n,i,j] > maxes[n]:
              maxes[n] = values[n,i,j]
            if values[n,i,j] < mins[n]:
              mins[n] = values[n,i,j]
    else:
      for j in range(0, jmax):
        for i in range(0, imax):
          values[n,i,j] = float(f.readline())
          if values[n,i,j] > maxes[n]:
            maxes[n] = values[n,i,j]
          if values[n,i,j] < mins[n]:
            mins[n] = values[n,i,j]

# Print message
  print('Successfully read data file ' + fname)

# Close the file
  f.close

  return (varcat, variables, values, mins, maxes)


def merge_metric(fileName, imax=128, jmax=128, kmax=1): 
    # raw data channels:
    # [0] freestream xmach
    # [1] freestream alpha
    # [2] freestream re
    # [3] rho  
    # [4] rhou 
    # [5] rhov 
    # [6] rhoE 
    print(imax, jmax)
    basename = fileName.split("_")[0]
    print(basename)

    #output_dir ="../Deep/data/test/"
    raw_fName=raw_dir+"/"+fileName
    npOutput_fName = output_dir+'/'+fileName
    npOutput = np.zeros((16, imax, jmax))

    raw_data=np.load(raw_fName)['a']
##
    ################################################################################
    metric_fname=train_metric_dir+"/"+basename+"_metric.p3d"

    _, variables, values, temp1, temp2, = read_function_file(metric_fname, imax, jmax, 1, False)
    imax, jmax, kmax, x, y, threed = read_grid("train_mesh/"+basename+".p3d")
    print(values.shape, npOutput.shape, variables, raw_data.shape)
    #rhoE = np.zeros((imax,jax)
    npOutput[0,:,:] = raw_data[0,:,:]
    npOutput[1,:,:] = raw_data[1,:,:]
    npOutput[2,:,:] = raw_data[2,:,:]


#    npOutput[3,:,:] = 0 #si4
#    npOutput[4,:,:] = 0 #sj1
#    npOutput[5,:,:] = 0 #sj3
#    npOutput[6,:,:] = 0
#    npOutput[7,:,:] = 0 #sk1
#    npOutput[8,:,:] = 0 #sk3
#    npOutput[9,:,:] = 0
#   
    npOutput[3,:,:] = values[0,:,:] #si4
    npOutput[4,:,:] = values[1,:,:] #sj1
    npOutput[5,:,:] = values[2,:,:] #sj3
    npOutput[6,:,:] = values[3,:,:] #sj4
    npOutput[7,:,:] = values[4,:,:] #sk1
    npOutput[8,:,:] = values[5,:,:] #sk3
    npOutput[9,:,:] = values[6,:,:] #sk4

    for i in range(128): 
        npOutput[14,:,i] = x[:,0]
        npOutput[15,:,i] = y[:,0]

    gamma = 1.4

    rho = raw_data[3,:,:] #* values[0,:,:] # rho/J
    npOutput[10,:,:] = rho
    npOutput[11,:,:] = raw_data[4,:,:]/raw_data[3,:,:] #* values[0,:,:] # rhou/J
    npOutput[12,:,:] = raw_data[5,:,:]/raw_data[3,:,:]  #* values[0,:,:] # rhov/J
    p = (raw_data[6,:,:] - 0.5*(raw_data[4,:,:]**2 + raw_data[5,:,:]**2)/raw_data[3,:,:]) * (gamma-1)
    npOutput[13,:,:] = np.power(gamma*p/rho, 0.5)
    ################################################################################
#dist_fname="../train_distance_p3d/"+basename+"_stats.p3d"

#    _, variables, values, temp1, temp2, = read_function_file(dist_fname, imax, jmax, 1, False)
#    print(values.shape, npOutput.shape, variables, raw_data.shape)
#    npOutput[3,:,:] = values[0,:,:] #distance refresh with distance

    #plt.imshow(values[0,:,:])
    #plt.imshow(npOutput[3,:,:])
    #plt.show() 

###
    #fileName = dataDir + str(uuid.uuid4()) # randomized name
    print(fileName)
    print("\tsaving in " + npOutput_fName + ".npz")
    np.savez_compressed(npOutput_fName, a=npOutput)





####### START from here #######

files = os.listdir(raw_dir)
files.sort()
if len(files)==0:
	print("error - no mesh file found" )
	exit(1)
print("Number of mesh files:", len(files))


imax=128
jmax=128
# main
for fileName in files:


    
    merge_metric(fileName, imax, jmax)
    print("\tdone")
