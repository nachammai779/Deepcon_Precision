import tensorflow as tf
import numpy as np
import datetime
import sys
import scipy.linalg as la 
import numpy.linalg as na

import keras.backend as K
epsilon = K.epsilon()
K.set_image_data_format('channels_last')
sys.setrecursionlimit(10000)
from tensorflow.python.lib.io import file_io

################################################################################
# Some GPU configs don't allow memory growth
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.tensorflow_backend.set_session(sess)

def ROPE(S, rho):
    p=S.shape[0]
    S=S
    try:
        LM=na.eigh(S)
    except:
        LM=la.eigh(S)
    L=LM[0]
    M=LM[1]
    for i in range(len(L)):
        if L[i]<0:
            L[i]=0
    lamda=2.0/(L+np.sqrt(np.power(L,2)+8*rho))
    indexlamda=np.argsort(-lamda)
    lamda=np.diag(-np.sort(-lamda)[:p])
    hattheta=np.dot(M[:,indexlamda],lamda)
    hattheta=np.dot(hattheta,M[:,indexlamda].transpose())
    return hattheta

def blockshaped(arr,dim=21):
    global threedarr
    p=arr.shape[0]//dim
    re=np.zeros([p,p,dim*dim])
    for i in range(p):
        for j in range(p):
            re[i,j,:]=arr[i*dim:i*dim+dim,j*dim:j*dim+dim].flatten()
    return re
    
def main(file1path, file2path, file3path, batch_list):
    MAP_CHANNELS = 60
    RAW_CHANNELS = 441
    rho = np.exp((np.arange(80)-60)/5.0)[30]
    lines = []
    f = open(batch_list, mode='r')
    lines = f.read()
    f.close()
    lines = lines.splitlines()

    with tf.device('/device:GPU:0'):
        for line in lines:
            print("Start " + line + " " + str(datetime.datetime.now()))
            fn=line.strip()
            rawdata = np.memmap(file1path+fn+'.map', dtype=np.float32, mode='r')
            ly = int(np.sqrt(rawdata.shape[0]/(MAP_CHANNELS+1)))
            x_ch_first = np.memmap(file2path+fn+'.21c', dtype=np.float32, mode='r+', shape=(1,RAW_CHANNELS,ly,ly))
            x = np.rollaxis(x_ch_first[0], 0, 3)
            xlen0 = x.shape[0]
            expand4D = x.reshape(xlen0,xlen0,21,21)
            fourd = np.moveaxis(expand4D, 2, 1)
            final2d = fourd.reshape(L*21,L*21)
            rho2 = np.exp((np.arange(80)-60)/5.0)[30]
            pre = ROPE(final2d,rho2)
            y = blockshaped(pre)
            
            fp = np.memmap(file3path+fn+'.pre21c', dtype=np.float32, mode='w+', shape=(ly,ly,RAW_CHANNELS))
            fp[:] = y[:]
            del rawdata
            del x_ch_first
            del fp
            print("End " + line + " " + str(datetime.datetime.now()))

file1path = "/ssd2tb/temp/map/"
file2path = "/nvme2tb/temp/21c-256/"
file3path = "/nvme2tb/temp/deepcon/pre21c/"
batch_list = "/nvme2tb/temp/train_ecod_1.lst"
main(file1path, file2path, file3path, batch_list)
