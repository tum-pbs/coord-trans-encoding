

import numpy as np
import math 
from matplotlib import pyplot as plt 

nlast = -0

cref = 1

folderName=input("Enter the folder name: ")
#folderName="history_files"
fileName=input("Enter the file name: ")
#fileName = "cfl3d.res"
#fileName = "cfl3d_ss.res"
#fileName = "cfl3d.subit_res"
ncyc=1
f=open(folderName+"/"+fileName,"r")
lines=f.readlines()
t = [] 
result=[]
cl=[]
cd=[]
cm=[]
i=1
for xline in lines:
       if i>8:
           tempt=(float(xline.split()[0]))
           if fileName=="cfl3d.subit_res":
               tempt/=ncyc
           t.append(tempt)
           
           result.append(float(xline.split()[1]))
           cl.append(float(xline.split()[2]))
           cd.append(float(xline.split()[3]))
           cm.append(float(xline.split()[-1]))
       i=i+1
f.close()
plt.figure(figsize = (20,15))
#plt.plot(np.asarray(t)*0.001,result)
#plt.subplot(2,2,1)
#plt.plot(np.asarray(t[-nlast:]),result[-nlast:],"k-")
#plt.title("Residual")
plt.subplot(3,2,1)
#plt.plot(cl[-nlast:],cd[-nlast:],"k-")
#plt.title(r"$C_l$")
plt.plot(t[-nlast:],result[-nlast:],"k-")
plt.plot(t[-nlast:],result[-nlast:],"ko")
plt.title(r"$Log-res$")

plt.subplot(3,2,2)
plt.plot(np.asarray(t[-nlast:]),cl[-nlast:],"k-")
plt.title(r"$C_l$")
plt.subplot(3,2,3)
plt.plot(np.asarray(t[-nlast:]),cd[-nlast:],"k-")
plt.title(r"$C_d$")
plt.subplot(3,2,4)
plt.plot(np.asarray(t[-nlast:]),cm[-nlast:],"k-")
plt.title(r"$C_m$")
plt.subplot(3,2,5)
plt.plot(np.asarray(cl[-nlast:]),cd[-nlast:],"k-")
plt.title(r"$C_l-C_d$")

plt.subplot(3,2,6)
plt.plot(np.asarray(cl[-nlast:]),cm[-nlast:],"k-")
plt.title(r"$C_l-C_m$")

#You can do the same using a list comprehension

#print [x.split(' ')[1] for xline in open(file).readlines()]
plt.show()


