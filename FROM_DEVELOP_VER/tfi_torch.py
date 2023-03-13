import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from global_variables import *


#from config import set_default_dtype
#set_default_dtype() 
#torch.set_default_dtype(torch.float64)
#from config import *

#################################### TFI #########################################
# 
# X = U + V - U*V
# U =  alpha_1^0 * X(xi_1, eta) + alpha_2^0 * X(xi_2, eta)
# alpha_1^0 = (1-xi)
# alpha_2^0 = xi
# V =  beta_1^0 * X(xi, eta_1) + beta_2^0 * X(xi, eta_2)
# beta_1^0 = (1-eta)
# beta_2^0 = eta
#  
#  
def calcTFIEdge(dP_te, dP_0, F, G, xi, eta, lojst, lojfn, upjst, upjfn):
    num_wake_points = lojfn-lojst+1




    dP_0_x  = dP_0[:,0].unsqueeze(1).expand(-1,num_wake_points)
    dP_0_y  = dP_0[:,1].unsqueeze(1).expand(-1,num_wake_points)
    dP_te_x = dP_te[:,0].unsqueeze(1).expand(-1,num_wake_points)
    dP_te_y = dP_te[:,1].unsqueeze(1).expand(-1,num_wake_points)

    denominator = F[:,0,lojfn-1,0].unsqueeze(1).expand(-1,num_wake_points)
    mylocal_F = torch.div(F[:,0,lojst-1:num_wake_points+lojst-1,0],denominator)
    
    test_x = (1.-mylocal_F)*dP_0_x + mylocal_F*dP_te_x
    test_y = (1.-mylocal_F)*dP_0_y + mylocal_F*dP_te_y
    dE_wake_lower = torch.cat((test_x.unsqueeze(1), test_y.unsqueeze(1)),dim=1).unsqueeze(-1)


    
    denominator = (F[:,0,upjfn-1,0]-F[:,0,upjst-1,0]).unsqueeze(1).expand(-1,num_wake_points)
    mylocal_F = (F[:,0,upjst-1:num_wake_points+upjst-1,0]-F[:,0,upjst-1,0].unsqueeze(1).expand(-1,num_wake_points))/denominator

    test_x = (1.-mylocal_F)*dP_te_x + mylocal_F*dP_0_x
    test_y = (1.-mylocal_F)*dP_te_y + mylocal_F*dP_0_y

    dE_wake_upper = torch.cat((test_x.unsqueeze(1), test_y.unsqueeze(1)),dim=1).unsqueeze(-1)
    
    
    
    
    

    return dE_wake_lower, dE_wake_upper

#def calcTFIEdge_old(dP_te, dP_0, F, G, xi, eta, lojst, lojfn, upjst, upjfn):
#    batch_size = dP_te.shape[0]
#    num_wake_points = lojfn-lojst+1
#    dE_wake_lower  = torch.zeros(batch_size, 2, num_wake_points, 1).cuda()
#    dE_wake_upper  = torch.zeros(batch_size, 2, num_wake_points, 1).cuda()
#    for ivar in range(2):
#        for i in range(0, num_wake_points):
#            #dE_wake_lower[ivar, i, 0] = (F[lojfn-1,0]-F[lojst-1+i,0])*dP_0[ivar]  + F[lojst-1+i,0]*dP_te[ivar] 
#            #dE_wake_upper[ivar, i, 0] = (F[upjfn-1,0]-F[upjst-1+i,0])*dP_te[ivar] + F[upjst-1+i,0]*dP_0[ivar] 
#            mylocal_F = F[:,0,lojst-1+i,0]/F[:,0,lojfn-1,0]
#            dE_wake_lower[:,ivar, i, 0] = (1.-mylocal_F)*dP_0[:,ivar]  + mylocal_F*dP_te[:,ivar] 
#            mylocal_F = (F[:,0,upjst-1+i,0]-F[:,0,upjst-1,0])/(F[:,0,upjfn-1,0]-F[:,0,upjst-1,0])
#            dE_wake_upper[:,ivar, i, 0] = (1.-mylocal_F)*dP_te[:,ivar] + mylocal_F*dP_0[:,ivar] 
#    return dE_wake_lower, dE_wake_upper
#
#def calcUV_old(dE, F, G, xi, eta):
#    batch_size = dE.shape[0]
#    print("I am in calcUV:", imax, jmax, batch_size)
#    Uij  = torch.zeros(batch_size, 2, imax, jmax).cuda().type(torch.cuda.DoubleTensor)
#    Vij  = torch.zeros(batch_size, 2, imax, jmax).cuda().type(torch.cuda.DoubleTensor)
#    for ivar in range(2):
#        for i in range(0,imax):
#            for j in range(0,jmax):
#                Uij[:,ivar,i,j] = (1- xi[:,0,i,j])*dE[:,ivar,0,0] +  xi[:,0,i,j]*dE[:,ivar,imax-1,0]
#                Vij[:,ivar,i,j] = (1-eta[:,0,i,j])*dE[:,ivar,i,0] + eta[:,0,i,j]*dE[:,ivar,i,0]  #???  .... 2023.02.26
#    return Uij, Vij



def calcUV(dE, F, G, xi, eta):
    batch_size = dE.shape[0]
    Uij  = torch.zeros(batch_size, 2, imax, jmax).cuda().type(torch.cuda.DoubleTensor)
    Vij  = torch.zeros(batch_size, 2, imax, jmax).cuda().type(torch.cuda.DoubleTensor)
    XI = xi[:, 0, :, :].unsqueeze(1).repeat(1, 2, 1, 1)
    ETA = eta[:, 0, :, :].unsqueeze(1).repeat(1, 2, 1, 1)
    #print("calcUV:",XI.shape, ETA.shape)
    #print("test dE:", dE[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1).shape)
    Uij[:,:,:,:] = (1 - XI) * dE[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1) \
                       + XI * dE[:, :, imax - 1, 0].unsqueeze(-1).unsqueeze(-1)
    #Vij[:,:,:,:] = (1 - ETA) * dE[:, :, :, 0].unsqueeze(-1) \
    #                   + ETA * dE[:, :, :, kkk].unsqueeze(-1)
    return Uij, Vij




def calcDeformation(dE, F, G, xi, eta, Uij, Vij):

    # assuming Uij, eta, and dE are already defined as torch tensors
    dS = torch.zeros(Uij.shape).cuda().type(torch.cuda.DoubleTensor)
    
    dS[:,:, :, :] = Uij[:, :, :, :]
    dS[:,:, :, :] += (1 - eta[:, 0, :, :].unsqueeze(1)) * (dE[:, :, :, 0].unsqueeze(3) - Uij[:, :, :, 0].unsqueeze(3))
    dS[:,:, :, :] += eta[:, 0, :, :].unsqueeze(1) * (-Uij[:, :, :, jmax-1].unsqueeze(3))


    return dS



#def calcDeformation_old(dE, F, G, xi, eta, Uij, Vij):
#    batch_size = dE.shape[0]
#    dS  = torch.zeros((batch_size, 2, imax, jmax), dtype=torch.float64).cuda()
#    for ivar in range(2):
#        for i in range(0,imax):
#            for j in range(0,jmax):
#                #dS[ivar,i,j]=Uij[ivar,i,j]+Vij[ivar,i,j]-(1-eta[i,j])*Uij[ivar,i,0]-eta[i,j]*Uij[ivar,i,jmax-1] # --- wrong!
#                dS[:,ivar,i,j]=Uij[:,ivar,i,j] + (1-eta[:,0,i,j])*(dE[:,ivar,i,0] - Uij[:,ivar,i,0]) + eta[:,0,i,j]*(-Uij[:,ivar,i,jmax-1]) # --- correct!
#                #dS[ivar,i,j]= (1-eta[i,j])*(dE[ivar,i,0]) 
#                #dS[ivar,i,j]= (1-eta[i,j])*(dE[ivar,i,0]) 
#                #dS[ivar,i,j]=(1-eta[i,j])*Uij[ivar,i,0]+eta[i,j]*Uij[ivar,i,jmax-1] # --- wrong!
#    return dS

def ComputeMeshInfo(x_original, y_original, xs_pred, ys_pred, jst, jfn, wake_lower_jst, wake_lower_jfn, wake_upper_jst, wake_upper_jfn):
    xs_initial = x_original[:,0,jst-1:jfn,0]
    ys_initial = y_original[:,0,jst-1:jfn,0]
    batch_size = x_original.shape[0]
    #print("......ys_initial and ys_pred:", ys_initial.shape, ys_pred.shape)

    _, _, _, _, _, _, _, f0, g0, Pij, xi, eta = calculateMetrics(x_original, y_original, FILL=True, CALC_ARC=True)

    dP_i0_j0   = torch.zeros((batch_size, 2), dtype=torch.float64).cuda() # this one doesn't change in the deformation
    dP_te      = torch.zeros((batch_size, 2), dtype=torch.float64).cuda()
    dP_te[:,0] = xs_pred[:,0] - xs_initial[:,0]
    dP_te[:,1] = ys_pred[:,0] - ys_initial[:,0]
    #print("TE point deformation:", dP_te.shape, ys_pred[0], ys_initial[0])

    #print(dP_te.shape, f0.shape, xi.shape)
    dE_wake_lower, dE_wake_upper = calcTFIEdge(dP_te, dP_i0_j0, f0, g0, xi, eta, wake_lower_jst, wake_lower_jfn, wake_upper_jst, wake_upper_jfn)
    dE_surface_x = (xs_pred - xs_initial).unsqueeze(1).unsqueeze(-1)
    dE_surface_y = (ys_pred - ys_initial).unsqueeze(1).unsqueeze(-1)

    #print("debug:", dE_wake_lower.shape, dE_wake_upper.shape)
    #print("debug dE_surface_x, dE_surface_y:", dE_surface_x.shape, dE_surface_y.shape)
    #print("after expand --> dE_surface_x, dE_surface_y:", dE_surface_x.shape, dE_surface_y.shape)



    #dE_surface = np.concatenate((dE_surface_x.reshape(1,jfn-jst+1,1), dE_surface_y.reshape(1,jfn-jst+1,1)), axis=0)
    #dE_surface = torch.cat((dE_surface_x.reshape(1,jfn-jst+1,1), dE_surface_y.reshape(1,jfn-jst+1,1)), 0)
    dE_surface = torch.cat([dE_surface_x, dE_surface_y], dim=1)
    #print("dE_surface:", dE_surface.shape)

    #dE = np.concatenate((dE_wake_lower[:,0:-1,:], dE_surface[:,0:-1,:], dE_wake_upper) ,axis=1)
    #dE = torch.cat((dE_wake_lower[:,0:-1,:], dE_surface[:,0:-1,:], dE_wake_upper), 1)
    dE = torch.cat((dE_wake_lower[:,:,0:-1,:], dE_surface[:,:,0:-1,:], dE_wake_upper), dim=2)
    #print("dE:", dE.shape)
    
    Uij, Vij = calcUV(dE, f0, g0, xi, eta)
    #print(Uij.shape, Vij.shape)
    dS = calcDeformation(dE, f0, g0, xi, eta, Uij, Vij)
    #print(dE.shape, dS.shape)

    return dS


def calculateMetrics(x, y, FILL=True, CALC_ARC=True, CHECK_METRICS=False):
  with torch.no_grad():
    '''
    Input x, y
    Outputs:
    mesh edge-i unit normal vector x-component: dx_i
    mesh edge-i unit normal vector y-component: dy_i
    mesh edge-i arc length:                     ds_i
    mesh edge-j unit normal vector x-component: dx_j
    mesh edge-j unit normal vector y-component: dy_j
    mesh edge-j arc length:                     ds_j
    '''
    #kernel_dx = torch.Tensor( [[-1,  0,  1],
    #                        [-2,  0,  2],
    #                        [-1,  0,  1]] ) * (1/8)
    #kernel_dy = torch.Tensor( [[-1, -2, -1],
    #                        [ 0,  0,  0],
    #                        [ 1,  2,  1]] ) * (1/8)
    bs = x.shape[0]
                                                                                                                                                                               
    kernel_i = torch.Tensor( [[-1.0], [1.0]]).cuda().type(torch.cuda.DoubleTensor)  # 2x1
    kernel_j = torch.Tensor( [[-1.0, 1.0]]  ).cuda().type(torch.cuda.DoubleTensor)    # 1x2
    kernel_i  = kernel_i.view(1,1,2,1)
    kernel_j  = kernel_j.view(1,1,1,2)
    # the first dim: # of input channels
    # the second dim: # of output channels
    # the thrid and fourth dims: kernel size
    #print(kernel_i, kernel_j)


    kernel_d1 = torch.Tensor( [[-1.0, 0.0], [ 0.0, 1.0]]).cuda().type(torch.cuda.DoubleTensor)  # 2x1
    kernel_d2 = torch.Tensor( [[ 0.0, 1.0], [-1.0, 0.0]]).cuda().type(torch.cuda.DoubleTensor)    # 1x2
    #print(kernel_d1)
    #print(kernel_d2)
    #print(kernel_d1.shape)
    #print(kernel_d2.shape)

    kernel_d1  = kernel_d1.view(1,1,2,2)
    kernel_d2  = kernel_d2.view(1,1,2,2)

    # calculate tangential direction
    
    ti1 = F.conv2d(x, kernel_i, padding=0)  # dx/di (or dx/dxi)
    ti2 = F.conv2d(y, kernel_i, padding=0)  # dy/di (or dy/dxi)
    #print(ti1.shape, ti2.shape)
    #print(ti1, ti2)
    ds_i = ((ti1)**2 + (ti2)**2)**0.5
    ## normalize
    #dx_i = torch.div(ti1, ds_i) 
    #dy_i = torch.div(ti2, ds_i)
    # rotate -90deg to get normal
    #dx_i = torch.div(-ti2, ds_i) 
    #dy_i = torch.div( ti1, ds_i)
    dx_i = -ti2/ds_i 
    dy_i =  ti1/ds_i

    #print(dx_i.shape, dy_i.shape)

    # calculate tangential direction
    tj1 = F.conv2d(x, kernel_j, padding=0) 
    tj2 = F.conv2d(y, kernel_j, padding=0) 
    ds_j = ((tj1)**2 + (tj2)**2)**0.5
    ## normalize
    #dx_j = torch.div(tj1, ds_j) 
    #dy_j = torch.div(tj2, ds_j) 
    # rotate 90deg to get normal
    #dx_j = torch.div( tj2, ds_j) 
    #dy_j = torch.div(-tj1, ds_j) 
    dx_j =  tj2/ds_j 
    dy_j = -tj1/ds_j 

    d1 = F.conv2d(x, kernel_d1, padding=0)**2 + F.conv2d(y, kernel_d1, padding=0)**2
    d2 = F.conv2d(x, kernel_d2, padding=0)**2 + F.conv2d(y, kernel_d2, padding=0)**2
    area = 0.5*torch.pow( d1*d2, 0.5)

    #print("ds_i, ds_j", ds_i.shape, ds_j.shape)
        
        
    if CALC_ARC:
        curve_length_i = torch.sum(ds_i, dim=2)#.view(1,jmax)
        curve_length_j = torch.sum(ds_j, dim=3)#.view(imax,1)

        #print("curve:", curve_length_i.shape, curve_length_j.shape) 
        # curve: torch.Size([1, 1, 129]) torch.Size([1, 1, 257])
 
        #arcF = torch.div(torch.cumsum(ds_i, dim=2), curve_length_i.expand(imax, jmax))
        #arcG = torch.div(torch.cumsum(ds_j, dim=3), curve_length_j.expand(imax, jmax))  
        arcF = torch.cumsum(ds_i, dim=2)
        arcG = torch.cumsum(ds_j, dim=3)  
        # torch.Size([1,1, 256, 129]) torch.Size([1,1, 257, 128])

        numToExpand = arcF.shape[2] #.....e.g. 256 
        curve_length_i_expanded = curve_length_i.unsqueeze(2).expand(-1, -1, numToExpand, -1)
        numToExpand = arcG.shape[3] # ....e.g. 128
        curve_length_j_expanded = curve_length_j.unsqueeze(3).expand(-1, -1, -1, numToExpand)
        #print("curve:", curve_length_i_expanded.shape, curve_length_j_expanded.shape) 

        arcF = arcF / curve_length_i_expanded
        arcG = arcG / curve_length_j_expanded
        # torch.Size([1,1, 256, 129]) torch.Size([1,1, 257, 128])
        arcF_i0 = torch.zeros_like(curve_length_i) # [:,1,129]
        arcF_i0 = arcF_i0.unsqueeze(2).expand(-1, -1, 1, -1)
        arcG_j0 = torch.zeros_like(curve_length_j) # [:,1,257]
        arcG_j0 = arcG_j0.unsqueeze(3).expand(-1, -1, -1, 1)
   
        arcF = torch.cat([arcF_i0, arcF], dim=2)
        arcG = torch.cat([arcG_j0, arcG], dim=3)




        temp_f = (arcF[:,:,:,jmax-1]-arcF[:,:,:,0]) # [:,1,257]
        temp_g = (arcG[:,:,imax-1,:]-arcG[:,:,0,:]) # [:,1,129]
        #print("temp_f and temp_g:", temp_f.shape, temp_g.shape) 
        #temp_f = temp_f.unsqueeze(3).expand(-1,-1,-1,jdim)
        #temp_g = temp_g.unsqueeze(2).expand(-1,-1,idim,-1)
        temp_f = temp_f.unsqueeze(3).expand(-1,-1,-1,jdim)
        temp_g = temp_g.unsqueeze(2).expand(-1,-1,idim,-1)

        Pij = 1 - temp_f*temp_g
        #print("Pij:", Pij.shape) 
        #    xi[i,j]  = (arcF[i,0]+arcG[0,j]*temp_f)/Pij[i,j]
        #    eta[i,j] = (arcG[0,j]+arcF[i,0]*temp_g)/Pij[i,j] 
        arcF_j0 = arcF[:,:,:,0]
        arcG_i0 = arcG[:,:,0,:]
        #print("arcF_j0 and arcG_i0:", arcF_j0.shape, arcG_i0.shape) 
        # arcF_j0 and arcG_i0: torch.Size([1, 1, 257]) torch.Size([1, 1, 129])

        arcF_j0 = arcF_j0.unsqueeze(3).expand(-1,-1,-1,jdim)
        arcG_i0 = arcG_i0.unsqueeze(2).expand(-1,-1,idim,-1)
        #print("after unsqueeze --> arcF_j0 and arcG_i0:", arcF_j0.shape, arcG_i0.shape) 
        #print("after unsqueeze --> temp_f and temp_g:", temp_f.shape, temp_g.shape) 
        xi  = (arcF_j0 + arcG_i0*temp_f)/Pij
        eta = (arcG_i0 + arcF_j0*temp_g)/Pij ################################################# 25.02.2023
        #eta = (arcG_i0)+0e-0 ################################################# 25.02.2023

    if FILL:
        slice_to_pad = dx_i[:,:,imax-2,:].unsqueeze(2).expand(-1,-1,1,-1)
        dx_i = torch.cat([dx_i, slice_to_pad], dim=2)
        slice_to_pad = dy_i[:,:,imax-2,:].unsqueeze(2).expand(-1,-1,1,-1)
        dy_i = torch.cat([dy_i, slice_to_pad], dim=2)
        slice_to_pad = ds_i[:,:,imax-2,:].unsqueeze(2).expand(-1,-1,1,-1)
        ds_i = torch.cat([ds_i, slice_to_pad], dim=2)

        slice_to_pad = dx_j[:,:,:,jmax-2].unsqueeze(3).expand(-1,-1,-1,1)
        dx_j = torch.cat([dx_j, slice_to_pad], dim=3)
        slice_to_pad = dy_j[:,:,:,jmax-2].unsqueeze(3).expand(-1,-1,-1,1)
        dy_j = torch.cat([dy_j, slice_to_pad], dim=3)
        slice_to_pad = ds_j[:,:,:,jmax-2].unsqueeze(3).expand(-1,-1,-1,1)
        ds_j = torch.cat([ds_j, slice_to_pad], dim=3)


        slice_to_pad = area[:,:,imax-2,:].unsqueeze(2).expand(-1,-1,1,-1)
        area = torch.cat([area, slice_to_pad], dim=2)
        #print("area:", area.shape)
        slice_to_pad = area[:,:,:,jmax-2].unsqueeze(3).expand(-1,-1,-1,1)
        area = torch.cat([area, slice_to_pad], dim=3)
        #print("area:", area.shape)

    if CHECK_METRICS:
        xInt_i = calculateInterface_i(x, x, y)
        yInt_i = calculateInterface_i(y, x, y)
        #x_cpu = x[0][0].cpu().detach().numpy()
        #y_cpu = y[0][0].cpu().detach().numpy()
        vmin = -1
        vmax =  1
        lv = np.r_[vmin: vmax: 29j*5]
        plt.subplot(1,2,1, aspect=1.)
        plt.contourf(x[0][0].cpu().detach().numpy(), y[0][0].cpu().detach().numpy(), dx_i[0][0].cpu().detach().numpy(), lv, vmin=vmin, vmax=vmax, cmap='jet')# 
        plt.colorbar()                                                                                                                                            
                                                                                                                                                                  
        lv = np.r_[vmin: vmax: 29j*5]                                                                                                                             
        plt.subplot(1,2,2, aspect=1.)                                                                                                                             
        plt.contourf(x[0][0].cpu().detach().numpy(), y[0][0].cpu().detach().numpy(), dy_i[0][0].cpu().detach().numpy(), lv, vmin=vmin, vmax=vmax, cmap='jet')# 
        plt.colorbar()

        plt.figure() 
        vmin = -1
        vmax =  1
        lv = np.r_[vmin: vmax: 29j*5]
        plt.subplot(1,2,1, aspect=1.)
        plt.contourf(x[0][0].cpu().detach().numpy(), y[0][0].cpu().detach().numpy(), dx_j[0][0].cpu().detach().numpy(), lv, vmin=vmin, vmax=vmax, cmap='jet')# 
        plt.colorbar()                                                                                                                                         
                                                                                                                                                               
        lv = np.r_[vmin: vmax: 29j*5]                                                                                                                          
        plt.subplot(1,2,2, aspect=1.)                                                                                                                          
        plt.contourf(x[0][0].cpu().detach().numpy(), y[0][0].cpu().detach().numpy(), dy_j[0][0].cpu().detach().numpy(), lv, vmin=vmin, vmax=vmax, cmap='jet')# 
        plt.colorbar()
        plt.show()



    if CALC_ARC:
        return dx_i, dy_i, ds_i, dx_j, dy_j, ds_j, area, arcF, arcG, Pij, xi, eta
    else:
        return dx_i, dy_i, ds_i, dx_j, dy_j, ds_j, area
    

def calculateInterface_i(q, x, y, FILL=True):
    '''
    Input q
    Outputs qc
    '''
    #kernel_dx = torch.Tensor( [[-1,  0,  1],
    #                        [-2,  0,  2],
    #                        [-1,  0,  1]] ) * (1/8)
    #kernel_dy = torch.Tensor( [[-1, -2, -1],
    #                        [ 0,  0,  0],
    #                        [ 1,  2,  1]] ) * (1/8)
    #kernel_i = torch.Tensor( [[.5], [.5]]).cuda().type(torch.cuda.FloatTensor)  # 2x1
    kernel_i = torch.Tensor( [[.5], [.5]]).cuda().type(torch.cuda.DoubleTensor)  # 2x1
    kernel_i = kernel_i.unsqueeze(0).unsqueeze(1).expand(1, 1, -1, -1)

    # Special note: Using a standard convolution where the filters are applied to all input channels. 
    # In this case, the kernel shape would be (out_channels=4, in_channels=4, kernel_size=2, kernel_size=1), 
    # which means that there are 4 filters, one for each output channel, and each filter has a size of 2x1 
    # and operates on all 4 input channels.
    # Thus, we decide to do one variable at one time. 
    # In a longer term, we could do it with torch.split into groups


    #print(kernel_i.shape)
    # 
    # output channel size, input channel size, kernel shape 2x1
    #
    # calculate tangential direction average
    
    #print(q.shape)
    qc = F.conv2d(q, kernel_i, padding=0)  
    #print(qc.shape)
    if FILL:
        slice_to_pad = qc[:,:,imax-2,:].unsqueeze(2).expand(-1,-1,1,-1)
        qc = torch.cat([qc, slice_to_pad], dim=2)

    if False:

        plt.figure() 
        vmin =  0.35
        vmax =  0.75
        lv = np.r_[vmin: vmax: 29j*5]
        plt.subplot(1,2,1, aspect=1.)
        plt.contourf(x[0][0].cpu().detach().numpy(), y[0][0].cpu().detach().numpy(), q[0][-1].cpu().detach().numpy(), lv, vmin=vmin, vmax=vmax, cmap='jet')# 
        plt.colorbar()                                                                                                                                       
                                                                                                                                                             
        lv = np.r_[vmin: vmax: 29j*5]                                                                                                                        
        plt.subplot(1,2,2, aspect=1.)                                                                                                                        
        plt.contourf(x[0][0].cpu().detach().numpy(), y[0][0].cpu().detach().numpy(), qc[0][-1].cpu().detach().numpy(), lv, vmin=vmin, vmax=vmax, cmap='jet')# 
        plt.colorbar()
        plt.show()
        
    return qc


def calculateCellCenter(q, x, y, FILL=True):
    '''
    Input q
    Outputs qc
    '''
    #kernel_dx = torch.Tensor( [[-1,  0,  1],
    #                        [-2,  0,  2],
    #                        [-1,  0,  1]] ) * (1/8)
    #kernel_dy = torch.Tensor( [[-1, -2, -1],
    #                        [ 0,  0,  0],
    #                        [ 1,  2,  1]] ) * (1/8)
    bs = q.shape[0]
    q = q.view(bs, -1, idim, jdim)
    #kernel_c = torch.Tensor( [[.25, .25], [.25, .25]]).cuda().type(torch.cuda.FloatTensor)  # 2x1
    kernel_c = torch.Tensor( [[.25, .25], [.25, .25]]).cuda().type(torch.cuda.DoubleTensor)  # 2x1
    kernel_c = kernel_c.unsqueeze(0).unsqueeze(1).expand(1, 1, -1, -1)

    # Special note: Using a standard convolution where the filters are applied to all input channels. 
    # In this case, the kernel shape would be (out_channels=4, in_channels=4, kernel_size=2, kernel_size=1), 
    # which means that there are 4 filters, one for each output channel, and each filter has a size of 2x1 
    # and operates on all 4 input channels.
    # Thus, we decide to do one variable at one time. 
    # In a longer term, we could do it with torch.split into groups


    #print(kernel_i.shape)
    # 
    # output channel size, input channel size, kernel shape 2x1
    #
    # calculate tangential direction average
    
    #print(q.shape)
    qc = F.conv2d(q, kernel_c, padding=0)  
    #print(qc.shape)
        
    return qc


