

# Python packages
import random, os, sys, datetime
import time  as sys_time
import numpy as np
from matplotlib import pyplot as plt
#import skfmm
#import  importlib.util
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './../')))
from scipy import spatial
from skimage.util.shape import view_as_windows
from scipy import interpolate
import pickle


# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.modules.loss as Loss 
import torch.nn.functional as F

# Helpers Import

from torch.utils.data         import DataLoader
#--------- Project Imports ----------#
from DfpNet             import TurbNetG
from dataset            import TurbDataset
import utils
from utils import log


plt.rcParams["font.family"] = "Arial"
#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})


kw = 0
jsta=25 #26 # node point 25 to 105
jend=105

# vor =  d(w)/dx - d(u)/dz  # in eta zeta
#     =  d(w)/deta * eta_x + d(w)/dzeta * zeta_x  
#      -(d(u)/deta * eta_z + d(u)/dzeta * zeta_z
#     =  (w[j,k] - w[j-1,k]) * sj1 + (w[j,k]-w[j,k-1]) * sk1
#       -(u[j,k] - u[j-1,k]) * sj3 - (u[j,k]-u[j,k-1]) * sk3
def cal_vorticity(sj1, sj3, sk1, sk3, u, w):
    jdim, kdim = 128, 128
    vorticity = np.copy(u)
    for j in range(1, jdim):
        for k in range(1, kdim):
            vorticity[j,k] =  (w[j,k] - w[j-1,k]) * sj1[j,k] + (w[j,k]-w[j,k-1]) * sk1[j,k] -(u[j,k] - u[j-1,k]) * sj3[j,k] - (u[j,k]-u[j,k-1]) * sk3[j,k]
        if k==0:
            vorticity[j,k] =  (w[j,k] - w[j-1,k]) * sj1[j,k] + (w[j,k+1]-w[j,k]) * sk1[j,k] -(u[j,k] - u[j-1,k]) * sj3[j,k] - (u[j,k+1]-u[j,k]) * sk3[j,k]
    if j==0:
        for k in range(1, kdim):
            vorticity[j,k] =  (w[j+1,k] - w[j,k]) * sj1[j,k] + (w[j,k]-w[j,k-1]) * sk1[j,k] -(u[j+1,k] - u[j,k]) * sj3[j,k] - (u[j,k]-u[j,k-1]) * sk3[j,k]
        if k==0:
            vorticity[j,k] =  (w[j+1,k] - w[j,k]) * sj1[j,k] + (w[j,k+1]-w[j,k]) * sk1[j,k] -(u[j+1,k] - u[j,k]) * sj3[j,k] - (u[j,k+1]-u[j,k]) * sk3[j,k]
    return vorticity



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


#files=[]
TEST_DEBUG=True
#liwei_plt_cm = plt.cm.magma
liwei_plt_cm = plt.cm.jet
#liwei_plt_cm = plt.cm.Spectral
error_plt_cm = plt.cm.magma
##folder="./test/"
#files = os.listdir(folder)
#files.sort()
#
suffix = "_" # customize loading & output if necessary
prefix = ""
folder="./BASIC_data_coordinates_final_metricsAll_1940/test_avg/"

#def main():
def cf_sens(fileName):
    start_time_main_loop = sys_time.time()
    #field_avg = np.zeros((4,128,128))
    field  = [] #np.zeros((6,4,128,128))
    vor = []
    p_drag = []
    p_lift = []
    v_drag = []
    v_lift = []
    #field_std = np.zeros((4,128,128))
    num_of_models = 0
    # ---------------------- Below only for sensitivity -------------#
    #for si in range(25):
    #    s = chr(96+si)
    #    if(si==0): 
    #        s = "" # check modelG, and modelG + char
    #    modelPath = "./"
    #    modelFn = modelPath + prefix + "modelG{}{}".format(suffix,s)
    #    if not os.path.isfile(modelFn):
    #        continue
    # ---------------------------------------------------------------#
    if True:
        modelPath = "./"
        modelFn = modelPath + "modelG"
        num_of_models+=1 
    
    
        #files = ["fx66h80_58_-24_1169.npz"] 
        files = [fileName] 
        
        #files.append("naca0012_50_221_945.npz")
    
    
        channelExpo=7
        cuda = torch.device('cuda') # Default CUDA device
        
        
        with open(modelPath+'/max_inputs.pickle', 'rb') as f: max_inputs = pickle.load(f)
        f.close()
        with open(modelPath+'/max_targets.pickle', 'rb') as f: max_targets = pickle.load(f)
        f.close()
        print("## max inputs  ##: ",max_inputs) 
        print("## max targets ##: ",max_targets) 
        with open(modelPath+'/min_inputs.pickle', 'rb') as f: min_inputs = pickle.load(f)
        f.close()
        with open(modelPath+'/min_targets.pickle', 'rb') as f: min_targets = pickle.load(f)
        f.close()
        print("## min inputs  ##: ",min_inputs) 
        print("## min targets ##: ",min_targets) 
        
        nsteps = 4
        max_targets = np.asarray(max_targets) 
        min_targets = np.asarray(min_targets) 
        ###############################################################################
        netG = TurbNetG(channelExponent=channelExpo)
        netG.load_state_dict( torch.load(modelFn, map_location='cpu') )
        #if torch.cuda.is_available:
        netG.cuda()
        netG.eval()
    #plt.figure()
    #for it in range(0, 1):
        for n in range(len(files)):
    #        n= it
            data=np.load(folder+files[n])['a']
            basename = files[n].split("_")[0]
            print(basename)
    #test_mesh_folder = "../../DATASET/test_mesh/"
    #        imax, jmax, kmax, x, y, threed = read_grid(test_mesh_folder+basename+".p3d")
            test_mesh_folder = "test_mesh/"
            imax, jmax, kmax, x, y, threed = read_grid(test_mesh_folder+basename+".p3d")
            xmach    = data[0,:,:]
            aoa    = data[1,:,:]
            re     = data[2,:,:]
            si4    = data[3]
            sj1    = data[4,:,:]
            sj3    = data[5,:,:]
            sj4    = data[6]
            sk1    = data[7,:,:]
            sk3    = data[8,:,:]
            sk4    = data[9]
            xin    = data[14,:,:]
            yin    = data[15,:,:]
    
            p_data = data[13,:,:]**2 * data[10,:,:] / 1.4
    
    
            xm_inf  = xmach[0,0]
            aoa_inf =   aoa[64,-1]
            re_inf  =  re[0,0]
            #rho_inf = np.mean(data[10,:,-1]) #rho
            #p_inf   = np.mean(p_data[:,-1])
            print("Reynolds & Mach number:",re_inf, xm_inf)
    
            mach_data = np.sqrt(data[11]**2+data[12]**2)/data[13]
            pt_data = p_data*(1+0.2*mach_data**2)**3.5
            p_inf   = np.sum(p_data[:,-1]*sk4[:,-1])/np.sum(sk4[:,-1])
            pt_inf   = np.sum(pt_data[:,-1]*sk4[:,-1])/np.sum(sk4[:,-1])
            rho_inf   = np.sum(data[10,:,-1]*sk4[:,-1])/np.sum(sk4[:,-1])
            print(p_inf, pt_inf)
            
    


        

            data[13,:,:] = p_data[:,:]
    
            #u_inf   = data[11,64,-1] #u
            #v_inf   = data[12,64,-1] #v
            a_inf   = np.mean(data[13,:,-1]) #a
      
            #xm_inf  = (u_inf**2+v_inf**2)**0.5/a_inf
            #re_inf  =    re[64,-1]
    
            xmach   = torch.from_numpy(xmach).type(torch.FloatTensor).cuda()
            aoa   = torch.from_numpy(aoa).type(torch.FloatTensor).cuda()
            re    = torch.from_numpy(re).type(torch.FloatTensor).cuda()
            si4    = torch.from_numpy(si4).type(torch.FloatTensor).cuda()
            sj1    = torch.from_numpy(sj1).type(torch.FloatTensor).cuda()
            sj3    = torch.from_numpy(sj3).type(torch.FloatTensor).cuda()
            sj4    = torch.from_numpy(sj4).type(torch.FloatTensor).cuda()
            sk1    = torch.from_numpy(sk1).type(torch.FloatTensor).cuda()
            sk3    = torch.from_numpy(sk3).type(torch.FloatTensor).cuda()
            sk4    = torch.from_numpy(sk4).type(torch.FloatTensor).cuda()
    
            xin   = torch.from_numpy(xin).type(torch.FloatTensor).cuda()
            yin   = torch.from_numpy(yin).type(torch.FloatTensor).cuda()
    
            xmach  =(xmach- min_inputs[0])/(max_inputs[0]-min_inputs[0]+1e-20)
            aoa  = (   aoa- min_inputs[1])/(max_inputs[1]-min_inputs[1])
            re =   (    re- min_inputs[2])/(max_inputs[2]-min_inputs[2]) #/100 #.... fsX=10
            si4 =    ( si4- min_inputs[3])/(max_inputs[3]-min_inputs[3]) #/100 #.... fsX=10
            sj1    =( sj1 - min_inputs[4])/(max_inputs[4]-min_inputs[4])
            sj3    =( sj3 - min_inputs[5])/(max_inputs[5]-min_inputs[5])
            sj4    =( sj4 - min_inputs[6])/(max_inputs[6]-min_inputs[6])
            sk1    =( sk1 - min_inputs[7])/(max_inputs[7]-min_inputs[7])
            sk3     =(sk3 - min_inputs[8])/(max_inputs[8]-min_inputs[8])
            sk4     =(sk4 - min_inputs[9])/(max_inputs[9]-min_inputs[9])
            xin     =(xin - min_inputs[10])/(max_inputs[10]-min_inputs[10])
            yin     =(yin - min_inputs[11])/(max_inputs[11]-min_inputs[11])
    
    
    
    
    
    
    
    #        print(max_inputs)
    #print(xmach[0,0].item(), aoa[0,0].item(), re[0,0].item())
    
    
    
            #input_gpu = torch.from_numpy(input).cuda()
    
            input_gpu = torch.cat( (xmach, aoa, re, si4, sj1, sj3, sj4, sk1, sk3, sk4, xin, yin) )
            input_gpu = input_gpu.view((1, 12, aoa.shape[0], aoa.shape[1]))
    
            #input = np.zeros((1, 3, binaryMaskInv.shape[0],binaryMaskInv.shape[1]))
            ##input[0, 0:,] = channelfsX
            ##input[0, 1:,] = channelfsY
            ##input[0, 2:,] = binaryMaskInv
            #input[0, 0,] = channelfsX
            #input[0, 1,] = channelfsY
            #input[0, 2,] = binaryMaskInv
               #input      = self._getInputArray(numpyMask)                                                                
            #plt.figure()
            #plt.imshow(channelfsX.detach().cpu().numpy()) #, levels)
            #plt.figure()
            #plt.imshow(channelfsY.detach().cpu().numpy()) #, levels)
            #plt.show()
            
    #        if TEST_DEBUG: 
    #            for i in range(3):
    #                plt.subplot(1,3, i+1)
    #                plt.imshow(input_gpu[0][i].detach().cpu().numpy()) 
    #                # as it is a variable that requires grad, need to detach(); also cannot convert a cuda var to numpy
    #                plt.colorbar()
    #            plt.show()
    
    
    
            #fullSolution = netG(torch.Tensor(input)).detach().numpy()
            fullSolution = netG(input_gpu)
    
            fullSolution = fullSolution.detach().cpu().numpy()
            for i in range(4):
                fullSolution[0][i] = fullSolution[0][i]*(max_targets[i]-min_targets[i])+min_targets[i]
    
            p_Solution   = fullSolution[0][3]**2 * (fullSolution[0][0]) / 1.4
            fullSolution[0][3] = p_Solution
       
            field.append(fullSolution[0])



            si4    = data[3]
            sj1    = data[4,:,:]
            sj3    = data[5,:,:]
            sj4    = data[6]
            sk1    = data[7,:,:]
            sk3    = data[8,:,:]
            sk4    = data[9]
            sj1    = sj1*sj4/si4
            sj3    = sj3*sj4/si4
            sk1    = sk1*sk4/si4
            sk3    = sk3*sk4/si4


            kw = 0
            jsta=25 #26 # node point 25 to 105
            jend=105


            jdim = 128 
            tangent_x = np.zeros(jdim)
            tangent_y = np.zeros(jdim)
            normal_x  = np.zeros(jdim)
            normal_y  = np.zeros(jdim)
            segment   = np.zeros(jdim)
            #for j in range(jsta, jend):
            for j in range(0, jdim-1):
                t1 = x[j+1,0]-x[j,0]
                t2 = y[j+1,0]-y[j,0]
                tangent_mag = t1**2+t2**2
                tangent_mag = tangent_mag**0.5
                tangent_x[j] = t1  / (1e-10+tangent_mag)
                tangent_y[j] = t2  / (1e-10+tangent_mag)
                segment[j]   = tangent_mag

                n1 = x[j,1]-x[j,0]
                n2 = y[j,1]-y[j,0]
                normal_mag = n1**2+n2**2
                normal_mag = normal_mag**0.5
                normal_x[j] = n1 / (1e-10+normal_mag)
                normal_y[j] = n2 / (1e-10+normal_mag)



            #t_x_Sol, t_y_Sol  = cal_tangent(fullSolution[0][1], fullSolution[0][2])  
            #t_x, t_y          = cal_tangent(data[11], data[12])                      
            #tau_w_Sol         = vorticitySolution * (-sk3 * t_x_Sol + sk1 * t_y_Sol)
            #tau_w             = vorticity         * (-sk3 * t_x     + sk1 * t_y    )
            #plt.plot(x[jsta:jend,kw], (fullSolution[0,3,jsta:jend,kw]-1/1.4)/0.5/xm_inf/xm_inf/rho_inf, "k-")
            #plt.plot(x[jsta:jend,kw], (          p_data[jsta:jend,kw]-1/1.4)/0.5/xm_inf/xm_inf/rho_inf, "g+")
            #plt.plot(x[jsta:jend,kw], (fullSolution[0,3,jsta:jend,kw]-p_inf)/0.5/xm_inf/xm_inf/rho_inf, "k-")
            #plt.plot(x[jsta:jend,kw], (          p_data[jsta:jend,kw]-p_inf)/0.5/xm_inf/xm_inf/rho_inf, "g+")
            #plt.plot(x[jsta:jend,kw], re_inf*    np.abs(vorticitySolution[jsta:jend,kw]                                    )/(pt_inf-p_inf), "k-")
            #plt.plot(x[jsta:jend,kw], re_inf*    np.abs(        vorticity[jsta:jend,kw]                                    )/(pt_inf-p_inf), "g+")
            #ax.plot(   x[jsta:jend,kw], xm_inf/re_inf/1e6*(tau_w_Sol[jsta:jend,kw])/(pt_inf-p_inf), "k-")
            #ax.scatter(x[jsta:jend,kw], xm_inf/re_inf/1e6*(    tau_w[jsta:jend,kw])/(pt_inf-p_inf), s=80, facecolors='none', edgecolors='r')
            kw=1

            freestream_x = np.cos(aoa_inf*np.pi/180)
            freestream_y = np.sin(aoa_inf*np.pi/180)
            print("aoa:", aoa_inf, freestream_x, freestream_y)   



            vor_instant = cal_vorticity(sj1, sj3, sk1, sk3, fullSolution[0][1], fullSolution[0][2])
            vor.append(vor_instant)



# VISCOUS:
            #drag_instant_x = np.sum(xm_inf/re_inf/1e6*vor_instant[jsta:jend,kw]*( -normal_y[jsta:jend]*tangent_x[jsta:jend]+normal_x[jsta:jend]*tangent_y[jsta:jend] )/(pt_inf-p_inf)     *segment[jsta:jend] * tangent_x[jsta:jend])
            #drag_instant_y = np.sum(xm_inf/re_inf/1e6*vor_instant[jsta:jend,kw]*( -normal_y[jsta:jend]*tangent_x[jsta:jend]+normal_x[jsta:jend]*tangent_y[jsta:jend] )/(pt_inf-p_inf)     *segment[jsta:jend] * tangent_y[jsta:jend])
            drag_instant_x = np.sum(xm_inf/re_inf/1e6*vor_instant[jsta:jend,kw]*( -normal_y[jsta:jend]*tangent_x[jsta:jend]+normal_x[jsta:jend]*tangent_y[jsta:jend] )/(0.5*xm_inf*xm_inf)     *segment[jsta:jend] * tangent_x[jsta:jend])
            drag_instant_y = np.sum(xm_inf/re_inf/1e6*vor_instant[jsta:jend,kw]*( -normal_y[jsta:jend]*tangent_x[jsta:jend]+normal_x[jsta:jend]*tangent_y[jsta:jend] )/(0.5*xm_inf*xm_inf)     *segment[jsta:jend] * tangent_y[jsta:jend])

            drag_instant = drag_instant_x * freestream_x + drag_instant_y * freestream_y
            lift_instant =-drag_instant_x * freestream_y + drag_instant_y * freestream_x
            
            v_drag.append(drag_instant)
            #lift.append(lift_instant)
# PRESSURE: https://en.wikipedia.org/wiki/Drag_coefficient
            drag_instant_x = np.sum( (-p_Solution[jsta:jend,kw]+p_inf) /(0.5*xm_inf*xm_inf)     *segment[jsta:jend] * normal_x[jsta:jend]) 
            drag_instant_y = np.sum( (-p_Solution[jsta:jend,kw]+p_inf) /(0.5*xm_inf*xm_inf)     *segment[jsta:jend] * normal_y[jsta:jend])
            #drag_instant_x = np.sum( (-p_Solution[jsta:jend,kw]) /(0.5*xm_inf*xm_inf)     *segment[jsta:jend] * normal_x[jsta:jend]) 
            #drag_instant_y = np.sum( (-p_Solution[jsta:jend,kw]) /(0.5*xm_inf*xm_inf)     *segment[jsta:jend] * normal_y[jsta:jend])

            drag_instant = drag_instant_x * freestream_x + drag_instant_y * freestream_y
            lift_instant =-drag_instant_x * freestream_y + drag_instant_y * freestream_x
            p_drag.append(drag_instant)

    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    #if False:
    #    i = 3
    #    vmin=np.min(data[i+10])
    #    vmax=np.max(data[i+10])
    #    lv = np.r_[vmin: vmax: 9j*5]
    #    plt.subplot(1,1, 1, aspect=1.)
    #    plt.contourf(x, y, data[i+10], lv, cmap=liwei_plt_cm)
    
    #    plt.xlim(-0.5,1.5)
    #    plt.ylim(-0.5,1.5)
    #    plt.colorbar(ticks=lv[0::nsteps])
    
    field = np.reshape(field, (num_of_models,4,128,128))
    vor = np.reshape(vor, (num_of_models,128,128))
    print(field.shape)
    print("num of models:", num_of_models) 


    field_avg = np.mean(field, axis=0)
    field_std = np.std(field,  axis=0)
    vor_avg = np.mean(vor, axis=0)
    vor_std = np.std(vor,  axis=0)
 

    #npOutput[4,:,:] = values[1]*values[3]/values[0] #sj1*sj4/vol
    #npOutput[5,:,:] = values[2]*values[3]/values[0] #sj3*sj4/vol
    #npOutput[7,:,:] = values[4]*values[6]/values[0] #sk1*sk4/vol
    #npOutput[8,:,:] = values[5]*values[6]/values[0] #sk3*sk4/vol



    si4    = data[3]
    sj1    = data[4,:,:]
    sj3    = data[5,:,:]
    sj4    = data[6]
    sk1    = data[7,:,:]
    sk3    = data[8,:,:]
    sk4    = data[9]
    sj1    = sj1*sj4/si4
    sj3    = sj3*sj4/si4
    sk1    = sk1*sk4/si4
    sk3    = sk3*sk4/si4
    




    # nxvor
    # ex  ey    ez
    # nx   0    nz    --- nx=sk1; nz=sk3
    #  0  omega  0

    # = [-omega*nz
    #            0
    #     omega*nx]
    
    # shear stress abs:
    #|| tau_w || = sqrt(omeaga**2 * nz**2 + omega**2)
    
    # um = 0.5*(u[:,kw]+u[:,kw+1]) 
    # wm = 0.5*(w[:,kw]+w[:,kw+1]) 
    # tangent_x = um/sqrt(um*um + wm*wm) 
    # tangent_z = wm/sqrt(um*um + wm*wm) 

    # tau_w = -omega*nz *tangent_x + omega*nx *tangent_y
    # tau_w = -omega*sk3*tangent_x + omega*sk1*tangent_y

    

    #vorticitySolution = cal_vorticity(sj1*sj4/si4, sj3*sj4/si4, sk1*sk4/si4, sk3*sk4/si4, fullSolution[0][1], fullSolution[0][2])
    #vorticity         = cal_vorticity(sj1*sj4/si4, sj3*sj4/si4, sk1*sk4/si4, sk3*sk4/si4, data[11], data[12])
    vorticitySolution = cal_vorticity(sj1, sj3, sk1, sk3, field_avg[1], field_avg[2])
    vorticity         = cal_vorticity(sj1, sj3, sk1, sk3, data[11], data[12])
    jdim = 128 
    tangent_x = np.zeros(jdim)
    tangent_y = np.zeros(jdim)
    normal_x  = np.zeros(jdim)
    normal_y  = np.zeros(jdim)
    segment   = np.zeros(jdim)
    #for j in range(jsta, jend):
    for j in range(0, jdim-1):
        t1 = x[j+1,0]-x[j,0]
        t2 = y[j+1,0]-y[j,0]
        tangent_mag = t1**2+t2**2
        tangent_mag = tangent_mag**0.5
        tangent_x[j] = t1  / (1e-10+tangent_mag)
        tangent_y[j] = t2  / (1e-10+tangent_mag)
        segment[j]   = tangent_mag

        n1 = x[j,1]-x[j,0]
        n2 = y[j,1]-y[j,0]
        normal_mag = n1**2+n2**2
        normal_mag = normal_mag**0.5
        normal_x[j] = n1 / (1e-10+normal_mag)
        normal_y[j] = n2 / (1e-10+normal_mag)



    #t_x_Sol, t_y_Sol  = cal_tangent(fullSolution[0][1], fullSolution[0][2])  
    #t_x, t_y          = cal_tangent(data[11], data[12])                      
    #tau_w_Sol         = vorticitySolution * (-sk3 * t_x_Sol + sk1 * t_y_Sol)
    #tau_w             = vorticity         * (-sk3 * t_x     + sk1 * t_y    )
    #plt.plot(x[jsta:jend,kw], (fullSolution[0,3,jsta:jend,kw]-1/1.4)/0.5/xm_inf/xm_inf/rho_inf, "k-")
    #plt.plot(x[jsta:jend,kw], (          p_data[jsta:jend,kw]-1/1.4)/0.5/xm_inf/xm_inf/rho_inf, "g+")
    #plt.plot(x[jsta:jend,kw], (fullSolution[0,3,jsta:jend,kw]-p_inf)/0.5/xm_inf/xm_inf/rho_inf, "k-")
    #plt.plot(x[jsta:jend,kw], (          p_data[jsta:jend,kw]-p_inf)/0.5/xm_inf/xm_inf/rho_inf, "g+")
    #plt.plot(x[jsta:jend,kw], re_inf*    np.abs(vorticitySolution[jsta:jend,kw]                                    )/(pt_inf-p_inf), "k-")
    #plt.plot(x[jsta:jend,kw], re_inf*    np.abs(        vorticity[jsta:jend,kw]                                    )/(pt_inf-p_inf), "g+")
    #ax.plot(   x[jsta:jend,kw], xm_inf/re_inf/1e6*(tau_w_Sol[jsta:jend,kw])/(pt_inf-p_inf), "k-")
    #ax.scatter(x[jsta:jend,kw], xm_inf/re_inf/1e6*(    tau_w[jsta:jend,kw])/(pt_inf-p_inf), s=80, facecolors='none', edgecolors='r')
    kw=1

#    ax.plot(   x[jsta:jend,kw], vorticitySolution[jsta:jend,kw], "k-")
#    ax.scatter(x[jsta:jend,kw],         vorticity[jsta:jend,kw], s=80, facecolors='none', edgecolors='r')


    freestream_x = np.cos(aoa_inf*np.pi/180)
    freestream_y = np.sin(aoa_inf*np.pi/180)
    print("*********************************")
    print("aoa:", aoa_inf, freestream_x, freestream_y)   
    p_drag = np.asarray(p_drag)
    v_drag = np.asarray(v_drag)

    drag = p_drag + v_drag
    print("total drag:",drag)
    print("mean drag:", np.mean(drag))
    print("mean Cd-p:", np.mean(p_drag), np.mean(p_drag)/np.mean(drag)*100,"%")
    print("mean Cd-v:", np.mean(v_drag), np.mean(v_drag)/np.mean(drag)*100,"%")
    #print(np.std(drag))
    print("std Cd-p:", np.std(p_drag))
    print("std Cd-v:", np.std(v_drag))
    print(np.std(p_drag)/np.mean(drag)*100,"% of Total")
    print(np.std(v_drag)/np.mean(drag)*100,"% of Total")
    print(np.std(p_drag)/np.mean(p_drag)*100,"% of p force")
    print(np.std(v_drag)/np.mean(v_drag)*100,"% of v force")
    #print("total lift:",lift)
    #print(np.mean(lift))
    #print(np.std(lift))
    #print(np.std(lift)/np.mean(lift)*100,"%")


    cf = xm_inf/re_inf/1e6*vor_avg[jsta:jend,kw]*( -normal_y[jsta:jend]*tangent_x[jsta:jend]+normal_x[jsta:jend]*tangent_y[jsta:jend] )/(pt_inf-p_inf)     *segment[jsta:jend] * tangent_x[jsta:jend]
    cf_std = xm_inf/re_inf/1e6*vor_std[jsta:jend,kw]*( -normal_y[jsta:jend]*tangent_x[jsta:jend]+normal_x[jsta:jend]*tangent_y[jsta:jend] )/(pt_inf-p_inf) *segment[jsta:jend] * tangent_x[jsta:jend]
    
    


    num_sigma = 1
    lower_bound = cf - cf_std*num_sigma
    upper_bound = cf + cf_std*num_sigma
    #plt.fill_between(x[jsta:jend,kw], lower_bound, upper_bound, facecolor="grey", alpha=0.5)
    plt.fill_between(x[jsta:jend,kw], lower_bound, upper_bound, facecolor="lightcoral") #, alpha=0.5)


    ax.plot(   x[jsta:jend,kw], cf, "r-", label="DNN")

    cf_reference = xm_inf/re_inf/1e6*vorticity[jsta:jend,kw]*( -normal_y[jsta:jend]*tangent_x[jsta:jend]+normal_x[jsta:jend]*tangent_y[jsta:jend]         )/(pt_inf-p_inf)*segment[jsta:jend] * tangent_x[jsta:jend]
    ax.scatter(x[jsta:jend,kw], cf_reference, s=40, facecolors='none', edgecolors='k', linewidths=0.5, label="CFD")
    print("*********************************")
    print("sanity check:")
    proj_surface = segment*normal_y 
    proj_surface = np.abs(segment*normal_y )



    wet_area = np.sum(proj_surface[jsta:jend])
    print( wet_area)

    print("*********************************")

    cf_integral = np.sum(cf)/wet_area
    cf_ref_integral = np.sum(cf_reference)/wet_area
    cf_std_integral = np.sum(cf_std)/wet_area
    print("cf int:", cf_integral)
    print("cf ref int:", cf_ref_integral)
    print("cf std int:", cf_std_integral)
    print("*********************************")


    #ax.plot(   x[jsta:jend,kw], xm_inf/re_inf/1e6*vorticitySolution[jsta:jend,kw]*( -normal_y[jsta:jend]*tangent_x[jsta:jend]+normal_x[jsta:jend]*tangent_y[jsta:jend] )/0.5/xm_inf/xm_inf/rho_inf, "k-")
    #ax.scatter(x[jsta:jend,kw], xm_inf/re_inf/1e6*vorticity[jsta:jend,kw]*( -normal_y[jsta:jend]*tangent_x[jsta:jend]+normal_x[jsta:jend]*tangent_y[jsta:jend]         )/0.5/xm_inf/xm_inf/rho_inf, s=80, facecolors='none', edgecolors='r')
#    ax.plot(   x[jsta:jend,kw], (sk3[jsta:jend,kw]), "k-")
#    ax.scatter(x[jsta:jend,kw], (sk1[jsta:jend,kw]), s=80, facecolors='none', edgecolors='r')
    #ax.plot(   x[jsta:jend,kw], re_inf*0.5*np.abs(vorticitySolution[jsta:jend,kw]+vorticitySolution[jsta-1:jend-1,kw])/(pt_inf-p_inf), "k-")
    #ax.scatter(x[jsta:jend,kw], re_inf*0.5*np.abs(        vorticity[jsta:jend,kw]+        vorticity[jsta-1:jend-1,kw])/(pt_inf-p_inf), s=80, facecolors='none', edgecolors='r')
    #ax.plot(x[jsta:jend,kw], re_inf*0.5*np.abs(        vorticity[jsta:jend,kw]+        vorticity[jsta-1:jend-1,kw])/(pt_inf-p_inf), "k+")
    ax.set_xlim(-0.1,1.1)
#    ax.set_ylim(-0.01, 0.015)
    ax.set_xlabel(r"$x/c$")
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax.set_ylabel(r"$C_f$")
    plt.tight_layout() 
    sup_name = files[n].split("_")
    ax.set_title(sup_name[0]+", Ma="+ str(float(sup_name[1])/100)+", AoA="+str(float(sup_name[2])/100)+r"$^{\circ}$, Re="+sup_name[3].split(".")[0]+"k")
    if "e221" in sup_name[0]:
        plt.legend()
    #plt.tight_layout() 
    
#    #u = fullSolution[0][1]
#    u = field_avg[1]
#    var_string = "x"
#    
#    vmin=np.min(u)
#    vmax=np.max(u)
#    lv = np.r_[vmin: vmax: 9j*5]
#    nsteps = 4
#
#
#

#    airfoil_x    = data[14,jsta:jend,0]
#    airfoil_y    = data[15,jsta:jend,0]
#
#
#    #ins = ax.inset_axes([0.7,0.8,0.2,0.2])
#    #ins = ax.inset_axes([0.0,0.0,1.,1.])
#    if aoa_inf > 0: 
#        ins = ax.inset_axes([0.4, 0.7, 0.4, 0.4])
#        #ins = ax.inset_axes([0.4, 0., 0.4, 0.4])
#    else:
#        ins = ax.inset_axes([0.4, 0.7, 0.4, 0.4])
#        #ins = ax.inset_axes([0.4, 0., 0.4, 0.4])
#         
#    ins.set_aspect(1.)
#    ins.plot(airfoil_x, airfoil_y, color='grey')
#
#    # Turn off tick labels
#    ins.set_yticklabels([])
#    ins.set_xticklabels([])
#    ins.axis('off')
    #plt.show()
#    if files[n][:4] == "e221":
#        ins = ax.inset_axes([0.32, 0.6, 0.4, 0.4])
#    else:
##        ins = ax.inset_axes([0.5, 0.6, 0.4, 0.4])
#        ins = ax.inset_axes([0.5, 0.6, 0.4, 0.4])
#         
#    #ins.set_aspect(1.)
#    ins.set_aspect(1.)
#    #ins.patch.set_facecolor('white')
#    ins.contourf(x, y, u, lv, cmap=liwei_plt_cm)
#    ins.set_xlim(-0.2,1.2)
#    if "e473" in sup_name[0]:
#        ins.set_ylim(-0.6,0.7)
#    else:
#        ins.set_ylim(-0.2,1.2)
#    
#    ins.axis('off')
#    ins.set_facecolor('#ffffff')





    #fig.savefig('./fig18/vort_'+files[n][:-4]+'.eps', format='eps', dpi=600)
    #fig.savefig('./results_test_skinfriction/vort_'+files[n][:-4]+'.png', format='png', dpi=600)
#plt.ylim(0.55,0.9)

    end_time_main_loop = sys_time.time()



    plt.show()
   



#print("Main loop execution time: {}".format(convertSecond(end_time_main_loop - start_time_main_loop)))
    #plt.figure()
    #plt.plot(time,history)
    #plt.savefig("./figures/loss_history.png")
if __name__ == '__main__':
    start = sys_time.time()
    files = os.listdir(folder)
    files.sort()
    #files = ["e221_71_343_661.npz"]
    #files = ["e473_55_29_1294.npz"]
    #files = ["fx84w097_78_420_1501.npz"]
    #files = ["mue139_68_-2_3178.npz"]


    files = [
    "e221_71_343_661.npz",
    "e473_55_29_1294.npz",
    "fx84w097_78_420_1501.npz",
    "mue139_68_-2_3178.npz"
    ]
    for fileName in files:
        cf_sens(fileName)
    end = sys_time.time()
    #logMsg("\tExecution time: {}".format(convertSecond(end - start)))

#print("Execution time: {}".format(convertSecond(end - start)))



