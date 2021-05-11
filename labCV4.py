# counturs computation via surface aprox. of local pixel window
import cv2
import numpy as np
import random
#from numba import jit
from numpy.lib.nanfunctions import nanprod
import time

def binarisation(image,treshhold):

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] < treshhold:
                image[i][j] = 0
            else:
                image[i][j] = 255
    return image

def gg(x1,x2): # for  get_surfase_coeffs
    g=np.matrix([x1,x2,1]) # not array, only martix!
    g=g.T
    return g

# surface approximation function of local pixel area
def get_surfase_coeffs(S =3, N1 = -1,N2 =1, M1 =-1,M2 =1): # in our case aproxx. surface is plane
    indN = 1 - N1
    indM = 1 - M1
    DiapN = range(N1,N2+1)
    DiapM = range(M1,M2+1)
    G = np.zeros([S,S])
    for idx in range(0,S):
        for n in DiapN:
            for m in DiapM:
                A = gg(n,m)[idx]
                B = gg(n,m).T

                G[idx,:] = G[idx,:]+np.matmul(A,B)

    Ginv = np.linalg.inv(G)
    F= np.zeros([N2+indN,M2+indM,S])
    for idx in range(0,S):
        for n in DiapN:
            for m in DiapM:
                for l in range(0,S):
                    A = Ginv[idx,l]
                    B = gg(n,m)[l]
                    idc = n+indN-1
                    idy = m+indM-1
                    F[idx,idy,idc]=F[idx,idy,idc]+A*B
    return F

def convolution(img,SEM):
    m,n= img.shape
 
    constant= (SEM.shape[0]-1)//2 # const 
    ImgConv= np.zeros((m,n), dtype=np.uint8) 
    for i in range(constant, m-constant):  
        for j in range(constant,n-constant):
            temp= img[i-constant:i+constant+1, j-constant:j+constant+1]
            product= temp*SEM
            #product = np.nan_to_num(product,nan=0)
            t= np.sum(product)
            ImgConv[i,j]= t
    return ImgConv





#There variants of structural elements for conv.

SEM1= np.array([[2,-1,2], [-1,-4,-1],[2,-1,2]], dtype=np.float)*(1/6) #Оператор Лапласа 

path='test.png'
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image2 = np.zeros(image.shape, dtype=np.uint8)
print('image shape', image.shape)
print('image2 shape', image2.shape)
start = time.time()
#image1 = sp_noise(image, 0.01)# add noize if need
# Дифференциальный метод 1-го порядка. Аппроксимация поверхностью 1-го порядка, окно 3x3 
mask_tensor = get_surfase_coeffs() #get tensor with shape 3x3x3 for convolution
image1 = convolution(image[:,:],mask_tensor[:,:,0]) + convolution(image[:,:],mask_tensor[:,:,1]) # simplified grad. computation
 # Применяем маску из оператора Лапласа         
#image1 = convolution(image[:,:],SEM1) 
image1 = binarisation(image1, 128)
print('mask tensor: ')
print(np.round(mask_tensor,3))

end = time.time()
print('Time of execution: ',end - start) # замер времени выполнения

# result visualisation.
while(1):
    cv2.imshow('Sourse',image)
    cv2.imshow('Result',image1)
    idx = cv2.waitKey(100)
    if idx==27:    # Esc key to stop
        break
