################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Generate training data via OpenFOAM
#
################

import os, math, uuid, sys, random
import numpy as np
#import utils 
import shutil
import subprocess
from matplotlib import pyplot as plt
freestream_angle  = 22.5 #degree  # -angle ... angle

cmesh_database  = "../mesh_generation/c-mesh/"
output_dir        = "./train/"
current_dir       = os.getcwd() 
#"/home/liwei/Simulations/cfl3d_TestCases/02_Airfoils/c_type_input_channels/"
seed = random.randint(0, 2**32 - 1)
np.random.seed(seed)
print("Seed: {}".format(seed))


def runSim(gridFile, xmach, alpha, re, ncyc, user_iteravg, user_rest, user_dt, user_ntstep):
    with open("input_template.inp", "rt") as inFile:
        with open("input_1.inp", "wt") as outFile:
            for line in inFile:
                #line = line.replace("user_xmach", "{}".format(xmach))
                line = line.replace("user_iteravg", str(int(user_iteravg)))
                line = line.replace("user_rest", str(int(user_rest)))
                line = line.replace("user_dt", "%10.5f"%user_dt)
                line = line.replace("user_ntstep", str(int(user_ntstep)))
                line = line.replace("user_gridFile", cmesh_database+"/"+gridFile+".bin")
                line = line.replace("user_xmach", "%10.5f"%xmach)
                line = line.replace("user_aoa", "%5.3f"%alpha)
                line = line.replace("user_re", "%5.3f"%re)
                line = line.replace("user_ncyc", str(int(ncyc)))
                outFile.write(line)

    return_value = os.system("/home/liwei/Codes/cfl3d/Hypersonichen_repo/CFL3D/build/cfl/seq/cfl3d_seq < input_1.inp")
    if return_value > 0:
        sys.exit("Problem when running cfl3d, stop.")

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


def outputProcessing(basename, xmach, alpha, re, dataDir=output_dir, imax=128, jmax=128, kmax=1, imageIndex=0): 
    # output layout channels:
    # [0] freestream xmach
    # [1] freestream alpha
    # [2] freestream re
    # [3] rhoE output
    # [4] rhou output
    # [5] rhov output
    print(imax, jmax)
    npOutput = np.zeros((7, imax, jmax))
##
    ################################################################################
    fname="plot3dg_stats.p3d"

    _, variables, values, temp1, temp2, = read_function_file(fname, imax, jmax, 1, False)
    print(values.shape, npOutput.shape, variables)
    #rhoE = np.zeros((imax,jax)
    npOutput[0,:,:] = xmach
    npOutput[1,:,:] = alpha
    npOutput[2,:,:] = re



    npOutput[3,:,:] = values[0,:,:] #rho
    npOutput[4,:,:] = values[1,:,:] #rhou
    npOutput[5,:,:] = values[2,:,:] #rhov
    npOutput[6,:,:] = values[3,:,:] #rhoE

    #plt.imshow(values[0,:,:])
    #plt.imshow(npOutput[3,:,:])
    #plt.show() 

###
    #fileName = dataDir + str(uuid.uuid4()) # randomized name
    fileName =  "%s_%d_%d_%d" % (basename, int(xmach*100), int(alpha*100), int(re*1000) )
    print(fileName)
    fileName = dataDir+'/'+fileName
    print("\tsaving in " + fileName + ".npz")
    np.savez_compressed(fileName, a=npOutput)





####### START from here #######

files = os.listdir(cmesh_database)
files.sort()
if len(files)==0:
	print("error - no mesh file found in %s" % cmesh_database)
	exit(1)
print("Number of mesh files:", len(files))


imax=128
jmax=128
# main

ista = 0
iend = len(files)
#iend = len(files)//10 # for distributed data generation
print(ista, iend)
for n in range(ista, iend):
    print("Run {}:".format(n))

    fileNumber = n # np.random.randint(0, len(files))
    basename = os.path.splitext( os.path.basename(files[fileNumber]) )[0]

    angle  = np.random.uniform(-0.5, 8) 
    xmach = np.random.uniform(0.55, 0.8)
    #re = np.random.uniform(3, 30) #million # note that construct2d default Re=1e6. Consider re-meshing when Re changes.
    re = np.random.uniform(0.5, 5) #million # note that construct2d default Re=1e6. Consider re-meshing when Re changes.
#    re = 1
#    angle = 2.5
#    xmach = 0.4

    print(basename)
    print("\tusing {}".format(files[fileNumber]))
####################################################################################################
    print("\tUsing M= %5.3f, angle %+5.3f, re= %+5.3f million" %(xmach, angle, re)  )
    user_iteravg = 0
    user_dt = -1.0 #5.0
    user_ntstep = 1
    user_rest = 0
    ncyc = 16000
    runSim(basename, xmach, angle, re, ncyc, user_iteravg, user_rest, user_dt, user_ntstep)

    #print("\tUsing M= %5.3f, angle %+5.3f, re= %+5.3f million" %(xmach, angle, re)  )
    #user_iteravg = 0
    #user_dt = -5.0
    #user_ntstep = 1
    #user_rest = 1
    #ncyc = 8000
    #runSim(basename, xmach, angle, re, ncyc, user_iteravg, user_rest, user_dt, user_ntstep)

    print("restart... start averaging...")
    user_iteravg = 1
    user_dt = -1.0 #5.0
    user_ntstep = 1
    user_rest = 1
    ncyc = 8000
    runSim(basename, xmach, angle, re, ncyc, user_iteravg, user_rest, user_dt, user_ntstep)
    
    #return_value = os.system(current_dir+"/plot3d_To_p3d") # at the moment, we don't change mesh
    return_value = os.system(current_dir+"/avg_To_p3d") # at the moment, we don't change mesh
    print(return_value)
    if return_value > 0:
        sys.exit("Problem when coverting cfl3d_avgq.p3d to plot3d_stats.p3d, stop.")
    else:
        outputProcessing(basename, xmach, angle, re, "./train_avg", imax, jmax, imageIndex=n)
        print("\tdone")
#Here we should use os mov file to do it not system command !!!!return_value = os.system(mv plot3dg.p3d current_dir+"/train_mesh/"+basename+".p3d") # at the moment, we don't change mesh


    return_value = os.system(current_dir+"/plot3d_To_p3d") # at the moment, we don't change mesh
    #return_value = os.system(current_dir+"/avg_To_p3d") # at the moment, we don't change mesh
    print(return_value)
    if return_value > 0:
        sys.exit("Problem when coverting plot3dq.bin to plot3d_stats.p3d, stop.")
    else:
        outputProcessing(basename, xmach, angle, re, "./train", imax, jmax, imageIndex=n)
        print("\tdone")
    shutil.copy("./plot3dg.p3d", "./train_mesh/"+basename+".p3d")
    os.remove("./plot3dg.p3d")
    os.remove("./plot3dg_stats.p3d")
    os.remove("./plot3dg.bin")
    os.remove("./plot3dq.bin")
    os.remove("./cfl3d_avgg.p3d")
    os.remove("./cfl3d_avgq.p3d")
#Here we should use os mov file to do it not system command !!!!return_value = os.system(mv plot3dg.p3d current_dir+"/train_mesh/"+basename+".p3d") # at the moment, we don't change mesh
    hist_fileName =  "%s_%d_%d_%d" % (basename, int(xmach*100), int(angle*100), int(re*1000) )
    shutil.copy("./cfl3d.res", "./history_files/"+hist_fileName+".res")
    shutil.copy("./restart.bin", "./run_files/"+hist_fileName+".bin")
    shutil.copy("./cfl3d.out", "./run_files/"+hist_fileName+".out")
    ##if not os.path.isfile("./train_mesh/"+basename+".p3d"):
    ##shutil.copy("./plot3dg.p3d", "./train_mesh/"+basename+".p3d")

