# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import codecs
import gzip
import lzma

import numpy as np
import torch
import torch.utils.data as data
from torchvision.datasets import MNIST
from torchvision import transforms

import lap



## reading mnist

mtrain=MNIST('_mnist', train=True, 
             transform=transforms.ToTensor(), 
             download=True)

## reading qmnist

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def open_maybe_compressed_file(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rb')
    elif path.endswith('.xz'):
        return lzma.open(path, 'rb')
    else:
        return open(path,'rb')
    
def read_idx2_int(path):
    with open_maybe_compressed_file(path) as f:
        data = f.read()
        assert get_int(data[:4]) == 12*256 + 2
        length = get_int(data[4:8])
        width = get_int(data[8:12])
        parsed = np.frombuffer(data, dtype=np.dtype('>i4'), offset=12)
        return torch.from_numpy(parsed.astype('i4')).view(length,width).long()

def read_idx3_ubyte(path):
    with open_maybe_compressed_file(path) as f:
        data = f.read()
        assert get_int(data[:4]) == 8 * 256 + 3
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)

qtrainimgs = read_idx3_ubyte("qmnist-train-images-idx3-ubyte")
qtrainlbls = read_idx2_int("qmnist-train-labels-idx2-int")
qtrain=data.TensorDataset(qtrainimgs.float()/255, qtrainlbls[:,0])

# let's go

def subset(dataset,digit):
    temp=[]
    for i in range(len(dataset)):
        (_,lbl) = dataset[i]
        if lbl == digit:
            temp += (i,)
    return data.Subset(dataset,temp)

def collect_images(data):
    with torch.no_grad():
        mimg = torch.FloatTensor(len(data),28*28)
        for i in range(len(data)):
            img, *_ = data[i]
            mimg[i].copy_(img.view(28*28))
        return mimg

def squared_distances(m,q):
    with torch.no_grad():
        mq = torch.mm(m,q.t())
        m2 = torch.norm(m,2,1).pow(2)
        q2 = torch.norm(q,2,1).pow(2)
        return m2[:,None] + q2[None,:] - 2 * mq

def squared_jittered_distances(m,q):
    with torch.no_grad():
        d = squared_distances(m,q)
        m3 = torch.reshape(m,(m.size(0),28,28))
        q3 = torch.reshape(q,(q.size(0),28,28))        
        mc = m3[:,1:27,1:27].reshape((m.size(0),26*26))
        for dx in range(3):
            for dy in range(3):
                if dy != 1 or dx != 1:                
                    qc = q3[:,dy:26+dy,dx:26+dx].reshape((q.size(0),26*26))
                    d = torch.min(d,squared_distances(mc,qc)+1)
        return d
    
def match_mnist_train_set():
    result = torch.DoubleTensor(len(mtrain),3)
    # map each digit with hungarian algorithm
    for digit in range(10):
        msubset = subset(mtrain,digit)
        qsubset = subset(qtrain,digit)
        print(f"Matching digit={digit} mlen={len(msubset)} qlen={len(qsubset)}")
        n = len(msubset)
        mimg = collect_images(msubset)
        qimg = collect_images(qsubset)
        dd = squared_jittered_distances(mimg,qimg)
        cost,x,_ = lap.lapjv(dd.numpy(), extend_cost=True)
        print(f"   cost={cost} len={n} average={cost/n}")
        for i in range(len(msubset)):
            mindice = msubset.indices[i]
            qindice = qsubset.indices[x[i]]
            c = dd[i,x[i]]
            result[mindice,0] = qindice
            result[mindice,1] = c
            result[mindice,2] = qtrainlbls[qindice,5]
    return result
    
def save_2dtensor(x,path):
    with open(path,'w') as f:
        print(".MAT 2",x.shape[0],x.shape[1], file=f)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                e = ' ' if j+1 < x.shape[1] else '\n'
                print(x[i,j].item(), end=e, file=f)


matches = match_mnist_train_set()
save_2dtensor(matches, "mnist_train_matches.txt")

