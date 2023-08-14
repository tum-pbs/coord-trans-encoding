import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

#from tfi_numpy import *
from tfi_torch import *
#extent=(-1,1,-1,1)
gamma = 1.4

idim = 129 #65
jdim = 129 #33

imax = 128
jmax = 128

jst=25               # as you may see, "jst" or "jfn" should have been named as "ist" or "ifn"            
jfn=105              # because they denote the circumferential direction around an airfoil

wake_lower_jst=1     # which is essentially "i"-direction in the current case
wake_lower_jfn=25    # however, due to the legacy codes from NASA, it is used here
wake_upper_jst=105   # 
wake_upper_jfn=129   # Nonetheless, here i is the circumferential, j is normal.

#########################################################################################################
# an example to create a surface mask: airfoilMask is based on points; 
# while segmentMask is based on cell faces-i
airfoilMask = torch.zeros(idim, jdim)
airfoilMask[jst-1:jfn,0] = 1.0
airfoilMask = airfoilMask.ge(0.5)


#segmentMask = torch.zeros(idim-1, jdim)
segmentMask = torch.zeros(idim, jdim) # for the FILL case
segmentMask[jst-1:jfn-1,0] = 1.0
segmentMask = segmentMask.ge(0.5)


cellcenterMask = torch.zeros(idim-1, jdim-1)
cellcenterMask[jst-1:jfn-1,0] = 1.0
cellcenterMask = cellcenterMask.ge(0.5)

airfoilMask = airfoilMask.cuda()
segmentMask = segmentMask.cuda()
cellcenterMask = cellcenterMask.cuda()
#########################################################################################################




folderName="train"
#folderName=input("Enter the folder name: ")
fileName=input("Enter the file name: ")

data=np.load(folderName+'/'+fileName)['a']
#data=np.load(fileName)['a']
basename=fileName.split("_")[0]
print(basename)


plt.subplot(241)
plt.imshow(data[0,:,:])
plt.colorbar()

plt.subplot(242)
plt.imshow(data[1,:,:])
plt.colorbar()

plt.subplot(243)
plt.imshow(data[2,:,:])
plt.colorbar()


print(data[0,0,0], data[1,0,0], data[2,0,0])
plt.subplot(244)
plt.imshow(data[3,:,:])
plt.colorbar()

plt.subplot(245)
plt.imshow(data[4,:,:])
plt.colorbar()

plt.subplot(246)
plt.imshow(data[5,:,:])
plt.colorbar()

plt.subplot(247)
plt.imshow(data[6,:,:])
plt.colorbar()

plt.show()




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




imax, jmax, kmax, xc, yc, threed = read_grid("train_mesh/"+basename+".p3d")

_, _, _, xp, yp, threed = read_grid("train_mesh_point/"+basename+".p3d")
#xc, yc =  convert_center(xp, yp)

batch_size = 1
x_current, y_current = torch.from_numpy(xp).cuda(), torch.from_numpy(yp).cuda()
x_current = x_current.view(batch_size,1,idim,jdim).type(torch.cuda.DoubleTensor)
y_current = y_current.view(batch_size,1,idim,jdim).type(torch.cuda.DoubleTensor)

# now calculate the coordinate transformation metrics according to the paper Comput and Fluids
# ideally this should be done with "cell-point-type" grid files.
#dx_i, dy_i, ds_i, dx_j, dy_j, ds_j = calculateMetrics(x_current, y_current)


dx_i, dy_i, ds_i, dx_j, dy_j, ds_j, area = calculateMetrics(x_current, y_current, FILL=True, CALC_ARC=False, CHECK_METRICS=True)
#dx_j_int = calculateInterface_i(dx_j, FILL=False)
#dy_j_int = calculateInterface_i(dy_j, FILL=False)
#ds_j_int = calculateInterface_i(dy_j, FILL=False)
#
#dx_i_int = calculateInterface_j(dx_i, FILL=False)
#dy_i_int = calculateInterface_j(dy_i, FILL=False)
#ds_i_int = calculateInterface_j(dy_i, FILL=False)

print("Shape of the metrics", dx_i.shape, dx_j.shape, area.shape)
#print("Shape of the metrics", dx_i_int.shape, dx_j_int.shape, area.shape)
#pressure_center = calculateCellCenter(dynamics[:,istep,3], x_current, y_current)



#pressure_center = calculateCellCenter(dynamics[:,istep,3], x_current, y_current)

# dx_i: the x-component of the unit normal-wise vector of i-face
# dy_i: the y-component of the unit normal-wise vector of i-face
# ds_i: the arc length of the cell along-i (circumferential)
# note: the face norm vec of i-face points into the airfoil (into the object)
 
# dx_j: the x-component of the unit normal-wise vector of j-face
# dy_j: the y-component of the unit normal-wise vector of j-face 
# ds_j: the arc length of the cell along-j (normal)
# note: the face norm vec of j-face points clock-wise direction


#xc = calculateCellCenter(x_current)
#yc = calculateCellCenter(y_current)
#x, y = xc[0,0].cpu().detach().numpy(), yc[0,0].cpu().detach().numpy()
x, y = xc, yc


if True:

    plt.figure() 
    vmin =  -1
    vmax =  1
    lv = np.r_[vmin: vmax: 29j*5]
    plt.subplot(1,2,1, aspect=1.)
    plt.contourf(xp, yp, dx_i[0,0].cpu().detach().numpy(), lv, vmin=vmin, vmax=vmax, cmap='jet')# 
    plt.colorbar()                                                                                                                                       
                                                                                                                                                         
    lv = np.r_[vmin: vmax: 29j*5]                                                                                                                        
    plt.subplot(1,2,2, aspect=1.)                                                                                                                        
    plt.contourf(xp, yp, dy_i[0,0].cpu().detach().numpy(), lv, vmin=vmin, vmax=vmax, cmap='jet')# 
    plt.colorbar()
    plt.show()







# data[0] - xmach # in our current case, 
# data[1] - aoa   # these three numbers are the 
# data[2] - re    # same in all the samples.
# data[3] - rho ... density
# data[4] - rho*u ... density times X-velocity
# data[5] - rho*v ... density times Y-velocity
# data[6] - rho*E ... density times total energy (it is complex, but below you will see how I calculate pressure)

print(ds_i.shape)


if True:
# now we check if the arc length is calculated correctly:
# the cell (python notation index from 0), so [jst-2] and [jfn-1] 
# are the same segment, which locates inmediately aft the trailing edge (the sharp point).
    ibatch, ichannel = 0, 0
    print("jst-2",ds_i[ibatch,ichannel,jst-2,0]) # ... on the lower wake interface
    print("jfn-1",ds_i[ibatch,ichannel,jfn-1,0]) # ... on the upper wake interface

    print("wake_lower_jfn - 1",ds_i[ibatch,ichannel,wake_lower_jst-1,0]) # ... on the upper wake interface
    print("wake_upper_jfn - 2",ds_i[ibatch,ichannel,wake_upper_jfn-2,0]) # ... on the lower wake interface
# Similarly, if we want to loop the segments along the airfoil surface, the index should range from
# jst-1 to jfn-2 (included), so in python e.g. [jst-1:jfn-1,0] 
    print("jst-1",ds_i[ibatch,ichannel,jst-1,0]) # the lower surface near the trailing edge
    print("jfn-2",ds_i[ibatch,ichannel,jfn-2,0]) # the upper surface near the trailing edge

# Well...if we want to loop the grid points along the airfoil surface, the index should range from
# jst-1 to jfn-1 (included), so in python e.g. [jst-1:jfn,0]
    print("jst-1",xp[jst-1,0],yp[jst-1,0])
    print("jfn-1",xp[jfn-1,0],yp[jfn-1,0])

u = data[4,:,:]/data[3,:,:]
v = data[5,:,:]/data[3,:,:]
# from rho*E, now calculate pressure)
p = (data[6,:,:] - 0.5*(data[4,:,:]**2 + data[5,:,:]**2)/data[3,:,:]) * (gamma-1)

sound = np.power(gamma*p / data[3,:,:], 0.5) # the speed of sound
mach  = np.power(u*u + v*v, 0.5) / sound     # this is the local Mach number, not the free-stream Mach number, so it is a variable rather than a constant


print(data[3].shape, x.shape)
plt.subplot(231, aspect=1.)
vmin = np.min(data[3])
vmax = np.max(data[3])
lv = np.r_[vmin: vmax: 9j*5]
plt.contourf(x, y, data[3], lv, cmap=plt.cm.twilight_shifted) #magma)
plt.xlim(-0.5,1.5)
plt.ylim(-0.5,1.5)
plt.title(r"$\rho$")
plt.colorbar()

vmin = np.min(u)
vmax = np.max(u)
lv = np.r_[vmin: vmax: 9j*5]
plt.subplot(232, aspect=1.)
plt.contourf(x, y, u, lv, cmap=plt.cm.twilight_shifted) #magma)
plt.xlim(-0.5,1.5)
plt.ylim(-0.5,1.5)
plt.title(r"$u$")
plt.colorbar()

vmin = np.min(v)
vmax = np.max(v)
lv = np.r_[vmin: vmax: 9j*5]
plt.subplot(233, aspect=1.)
plt.contourf(x, y, v, lv, cmap=plt.cm.twilight_shifted) #magma)
plt.xlim(-0.5,1.5)
plt.ylim(-0.5,1.5)
plt.title(r"$v$")
plt.colorbar()

vmin = np.min(p)
vmax = np.max(p)
lv = np.r_[vmin: vmax: 9j*5]
plt.subplot(234, aspect=1.)
plt.contourf(x, y, p, lv, cmap=plt.cm.twilight_shifted) #magma)
plt.xlim(-0.5,1.5)
plt.ylim(-0.5,1.5)
plt.title(r"$p$")
plt.colorbar()

vmin = np.min(sound)
vmax = np.max(sound)
lv = np.r_[vmin: vmax: 9j*5]
plt.subplot(235, aspect=1.)
plt.contourf(x, y, sound, lv, cmap=plt.cm.twilight_shifted) #magma)
plt.xlim(-0.5,1.5)
plt.ylim(-0.5,1.5)
plt.title(r"$a$")
plt.colorbar()

vmin = .0
vmax = np.max(mach)
lv = np.r_[vmin: vmax: 9j*5]
plt.subplot(236, aspect=1.)
plt.contourf(x, y, mach, lv, cmap=plt.cm.twilight_shifted) #magma)
plt.xlim(-0.5,1.5)
plt.ylim(-0.5,1.5)
plt.title(r"$Mach$")
plt.colorbar()

plt.show()




angle_of_attack = 1.25 #degree
angle_of_attack = angle_of_attack / 180. * np.pi
cosDir = np.cos(angle_of_attack)
sinDir = np.sin(angle_of_attack)


pressure_center = torch.from_numpy(p).cuda().type(torch.cuda.DoubleTensor)   # in this case, flowfield variables are stored at cell center
#pressure_farfield_boundary = torch.from_numpy(p[:,-1]).cuda().type(torch.cuda.DoubleTensor)
#pressure_farfield_boundary = pressure_farfield_boundary.view(imax,1)
#print(pressure_farfield_boundary.shape)
#pressure_center = torch.cat((pressure_center, pressure_farfield_boundary), dim=1) # just make sure it has the same shape as ds and dx
print(pressure_center.shape)

xc_cuda = torch.from_numpy(xc).cuda().type(torch.cuda.DoubleTensor)
#yc_cuda = torch.from_numpy(yc).cuda().type(torch.cuda.DoubleTensor)
pressure_surface = torch.masked_select( pressure_center, cellcenterMask ).view(1,-1)
xc_surface       = torch.masked_select( xc_cuda,         cellcenterMask ).view(1,-1)
arcLength_surface = torch.masked_select( ds_i,           segmentMask ).view(1,-1)  # for the "FILL" case, it is actually at cell or "segment" not at point
nx_surface = torch.masked_select( -dx_i,           segmentMask ).view(1,-1)        # for the "FILL" case, it is actually at cell or "segment" not at point
ny_surface = torch.masked_select( -dy_i,           segmentMask ).view(1,-1)        # for the "FILL" case, it is actually at cell or "segment" not at point


#plt.figure()
fig, axs = plt.subplots(2, 2)
axs[0,0].plot(xc_surface[0].cpu().detach().numpy(),  pressure_surface[0].cpu().detach().numpy())
axs[0,1].plot(xc_surface[0].cpu().detach().numpy(), arcLength_surface[0].cpu().detach().numpy())
axs[1,0].plot(xc_surface[0].cpu().detach().numpy(), nx_surface[0].cpu().detach().numpy())
axs[1,1].plot(xc_surface[0].cpu().detach().numpy(), ny_surface[0].cpu().detach().numpy())
plt.show()



dfx = (1.4*pressure_surface-1)*arcLength_surface*nx_surface/(0.5*0.8**2*1.4)
dfy = (1.4*pressure_surface-1)*arcLength_surface*ny_surface/(0.5*0.8**2*1.4)

fx =     torch.sum( dfx, dim=1 ) 
fy =     torch.sum( dfy, dim=1 ) 
# for sanity check, calculate the wet surface, theoretically it should 2.0
wet_surface =     torch.sum( arcLength_surface, dim=1 ) 

Cd = fx*cosDir + fy*sinDir
Cl =-fx*sinDir + fy*cosDir


print(Cl.item(), Cd.item(), wet_surface.item())
