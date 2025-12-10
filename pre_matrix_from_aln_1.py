import numpy as np
import datetime
import scipy.linalg as la 
import numpy.linalg as na
import os
from subprocess import Popen, PIPE, STDOUT
from io import BytesIO
aadic={
        'A':1,
        'B':0,
        'C':2,
        'D':3,
        'E':4,
        'F':5,
        'G':6,
        'H':7,
        'I':8,
        'J':0,
        'K':9,
        'L':10,
        'M':11,
        'N':12,
        'O':0, 
        'P':13,
        'Q':14,
        'R':15,
        'S':16,
        'T':17,
        'U':0,
        'V':18,
        'W':19,
        'X':0,
        'Y':20,
        'Z':0,
        '-':0,
        '*':0,
        }

def readsequence(seq_file):
    lines=open(seq_file).readlines()
    lines=[line.strip() for line in lines]
    seq=lines[1].strip()  
    aalines=''
    for i in range(1,len(lines)):
        aalines+=lines[i]
    aas=[aadic[aa] for aa in aalines]
    return aas

def read_msa(file_path):    
    lines=open(file_path).readlines()  
    lines=[line.strip() for line in lines]   
    n=len(lines)
    d=len(lines[0]) #CR AND LF  
    msa=np.zeros([n,d],dtype=int)
    for i in  range(n):
        aline=lines[i]
        for j in range(d):
            msa[i,j]=aadic[aline[j]]
    return msa

def cal_large_matrix1(msa,weight):
    #output:21*l*21*l
    ALPHA=21
    pseudoc=1
    M=msa.shape[0]
    N=msa.shape[1]
    pab=np.zeros((ALPHA,ALPHA))
    pa=np.zeros((N,ALPHA))
    cov=np.zeros([N*ALPHA,N*ALPHA ])
    for i in range(N):
        for aa in range(ALPHA):
            pa[i,aa] = pseudoc
        neff=0.0
        for k in range(M):
            pa[i,msa[k,i]]+=weight[k]
            neff+=weight[k]
        for aa in range(ALPHA):
            pa[i,aa] /=pseudoc * ALPHA * 1.0 + neff
    #print(pab)
    for i in range(N):
        for j in range(i,N):
            for a in range(ALPHA):
                for b in range(ALPHA):
                    if i ==j :
                        if a==b :
                            pab[a,b]=pa[i,a]
                        else:
                            pab[a,b]=0.0
                    else:
                        pab[a,b] = pseudoc *1.0 /ALPHA
            if(i!=j):
                neff2=0;
                for k in range(M):
                    a=msa[k,i]
                    b=msa[k,j]
                    tmp=weight[k]
                    pab[a,b]+=tmp
                    neff2+=tmp
                for a in range(ALPHA):
                    for b in range(ALPHA):
                        pab[a,b] /= pseudoc*ALPHA*1.0 +neff2
            for a in range(ALPHA):
                for b in range(ALPHA):
                    if(i!=j or a==b):
                        if (pab[a][b] > 0.0):
                            cov[i*21+a][j*21+b]=pab[a][b] - pa[i][a] * pa[j][b]
                            cov[j*21+b][i*21+a]=cov[i*21+a][j*21+b]
    return cov


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
    p=arr.shape[0]//dim
    re=np.zeros([dim*dim,p,p])
    for i in range(p):
        for j in range(p):
            re[:,i,j]=arr[i*dim:i*dim+dim,j*dim:j*dim+dim].flatten()
    return re

def computepre(msafile,weightfile):
    msa=read_msa(msafile)
    weights=np.genfromtxt(weightfile).flatten()
    cov=cal_large_matrix1(msa,weights)
    rho2=np.exp((np.arange(80)-60)/5.0)[30]
    pre=ROPE(cov,rho2)
    L=int(cov.shape[0]/21)
    #print(pre)
    return L, blockshaped(pre)

def getweights_out(msafile,seq_id,outfile):
    exefile='/home/nachammai/calNf_ly'
    cmd=exefile+' '+msafile+' '+str(seq_id)+' >'+outfile
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,close_fds=True)
    output,error = p.communicate()

def processfile(file1path, file2path, line):
    print("Start " + line + str(datetime.datetime.now()))
    MAP_CHANNELS = 60
    RAW_CHANNELS = 441
    rho = np.exp((np.arange(80)-60)/5.0)[30]
    fn=line.strip()

    aln = file1path+fn+'.aln'
    savefile = file2path+fn
    
    seq_id=0.8
    weightfile=savefile+'.weight'
    getweights_out(aln,seq_id,weightfile)
        
    L, y = computepre(aln,weightfile)
    
    fp = np.memmap(savefile+'.pre21c', dtype=np.float32, mode='w+', shape=(RAW_CHANNELS,L,L))
    fp[:] = y[:]
    del fp
    print("End " + line + str(datetime.datetime.now()))    

def main(file1path, file2path, batch_list):
    lines = []
    f = open(batch_list, mode='r')
    lines = f.read()
    f.close()
    lines = lines.splitlines()

    for line in lines:
        processfile(file1path, file2path, line)

file1path = "/ssd2tb/temp/aln/"
file2path = "/ssd2tb/temp/pre21c/"
batch_list = "/home/nachammai/train_ecod_1.lst"
main(file1path, file2path, batch_list)
