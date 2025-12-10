#!/usr/bin/env python
# coding: utf-8

# In[37]:


from keras.layers import *
from keras.models import Model
import os, sys, datetime
import tensorflow as tf
import numpy as np
K.set_image_data_format('channels_last')
import argparse
import scipy.linalg as la 
import numpy.linalg as na
global L, zd, ze, zg, zh, zi


# In[38]:


n_channels = 441
pathname = os.path.dirname(sys.argv[0])
model_weights_file_name = pathname + '\weights-rdd-covariance.hdf5'


# In[39]:


def aanum(ch):
    aacvs = [999, 0, 3, 4, 3, 6, 13, 7, 8, 9, 21, 11, 10, 12, 2,
            21, 14, 5, 1, 15, 16, 21, 19, 17, 21, 18, 6]
    if ch.isalpha():
        return aacvs[ord(ch) & 31]
    return 20


# In[40]:


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


# In[41]:


# Python reimplementation of the original code by David Jones @ UCL
def cov21stats(file_aln):
    global L, zd, ze, zg, zh, zi
    alnfile = np.loadtxt(file_aln, dtype = str)
    nseq = len(alnfile)
    L = len(alnfile[0])
    aln = np.zeros(((nseq, L)), dtype = int)
    for i, seq in enumerate(alnfile):
        for j in range(L):
            aln[i, j] = aanum(seq[j])
     # Calculate sequence weights
    idthresh = 0.38
    weight = np.ones(nseq)
    for i in range(nseq):
        for j in range(i+1, nseq):
            nthresh = int(idthresh * L)
            for k in range(L):
                if nthresh > 0:
                    if aln[i, k] != aln[j, k]:
                        nthresh = nthresh - 1
            if nthresh > 0:
                weight[i] += 1
                weight[j] += 1
    weight = 1/weight
    wtsum  = np.sum(weight)
    # Calculate singlet frequencies with pseudocount = 1
    pa = np.ones((L, 21))
    for i in range(L):
        for a in range(21):
            pa[i, a] = 1.0
        for k in range(nseq):
            a = aln[k, i]
            if a < 21:
                pa[i, a] = pa[i, a] + weight[k]
        for a in range(21):
            pa[i, a] = pa[i, a] / (21.0 + wtsum)
    # Calculate pair frequencies with pseudocount = 1
    pab = np.zeros((L, L, 21, 21))
    for i in range(L):
        for j in range(L):
            for a in range(21):
                for b in range(21):
                    pab[i, j, a, b] = 1.0 / 21.0
            for k in range(nseq):
                a = aln[k, i]
                b = aln[k, j]
                if (a < 21 and b < 21):
                    pab[i, j, a, b] = pab[i, j, a, b] + weight[k];
            for a in range(21):
                for b in range(21):
                    pab[i, j, a, b] = pab[i, j, a, b] / (21.0 + wtsum)
    final = np.zeros((L, L, 21, 21))
    for a in range(21):
        for b in range(21):
            for i in range(L):
                for j in range(L):
                    final[i, j, a, b] = pab[i, j, a, b] - pa[i, a] * pa[j, b]    
    #4D to 2D
    zd = np.moveaxis(final, 2, 1)
    ze = zd.reshape(L*21,L*21)

    # Apply ROPE
    
    rho2=np.exp((np.arange(80)-60)/5.0)[30]
    pre=ROPE(ze,rho2)
    
    # Postprocess to channels_last
    
    zg = pre.reshape(L, 21, L, 21)
    zh = np.moveaxis(zg, 1, 2)
    zi = zh.reshape(L, L, 21 * 21)
    
    final4d = zi.reshape(1, L, L, 21 * 21)
    return final4d


# In[42]:


def main(aln, file_rr):
    print("Start " + str(sys.argv[0]) + " - " + str(datetime.datetime.now()))
    global n_channels
    global model_weights_file_name
    if not os.path.isfile(model_weights_file_name):
        print('Model weights file ' + model_weights_file_name + ' is absent!\n')
        print('Please download from https://github.com/badriadhikari/DEEPCON/')
        sys.exit(1)
    sequence = ''
    print ('')
    print ('Read sequence[0] from aln..')
    with open(aln) as f:
        sequence = f.readline()
        sequence = sequence.strip()
    L = len(sequence)
    print(L)
    #if L < 20:
    if L < 3:
        print ("ERROR!! Too short sequence!!")
    print ('')
    print ('Convert aln to inverse covariance matrix.. patience..')
    sys.stdout.flush()
    X = cov21stats(aln)
    if X.shape != (1, L, L, n_channels):
        print('Unexpected shape from cov21stats!')
        print(X.shape)
        sys.exit(1)
    print ('')
    print ('Build a model of the size of the input (and not bigger)..')
    sys.stdout.flush()
    dropout_value = 0.3
    input_original = Input(shape = (L, L, n_channels))
    tower = input_original
    
# Start - Maxout Layer for DeepCov dataset
    input = BatchNormalization()(input_original)
    input = Activation('relu')(input)
    input = Convolution2D(128, 1, padding = 'same')(input)
    input = Reshape((L, L, 128, 1))(input)
    input = MaxPooling3D(pool_size = (1, 1, 2))(input)
    input = Reshape((L, L, 64 ))(input)
    tower = input
    # End - Maxout Layer
    n_channels = 64
    d_rate = 1
    for i in range(32):
        block = BatchNormalization()(tower)
        block = Activation('relu')(block)
        block = Convolution2D(64, 3, padding = 'same')(block)
        block = Dropout(dropout_value)(block)
        block = Activation('relu')(block)
        block = Convolution2D(n_channels, 3, dilation_rate=(d_rate, d_rate), padding = 'same')(block)
        tower = add([block, tower])
        if d_rate == 1:
            d_rate = 2
        elif d_rate == 2:
            d_rate = 4
        else:
            d_rate = 1
    tower = BatchNormalization()(tower)
    tower = Activation('relu')(tower)
    tower = Convolution2D(1, 3, padding = 'same')(tower)
    output = Activation('sigmoid')(tower)
    model = Model(input_original, output)
    print ('')
    print ('Load weights from ' + model_weights_file_name + '..')
    model.load_weights(model_weights_file_name)
    print ('')
    print ('Predict..')
    sys.stdout.flush()
    P1 = model.predict(X)
    P2 = P1[0, 0:L, 0:L]
    P3 = np.zeros((L, L))
    for p in range(0, L):
        for q in range(0, L):
            P3[q, p] = (P2[q, p] + P2[p, q]) / 2.0
    print ('')
    print ('Write RR file ' + file_rr + '.. ')
    rr = open(file_rr, 'w')
    rr.write(sequence + "\n")
    for i in range(0, L):
        for j in range(i, L):
            if abs(i - j) < 5:
                continue
            rr.write("%i %i 0 8 %.5f\n" %(i+1, j+1, P3[i][j]))
    rr.close()
    print("Done " + str(sys.argv[0]) + " - " + str(datetime.datetime.now()))


# In[43]:


aln = r'C:\Users\Nachammai\Desktop\Thesis\16pkA0.aln.txt'
file_rr = r'C:\Users\Nachammai\Desktop\Thesis\deepcon_pre_op_0919.txt'
main(aln,file_rr)

