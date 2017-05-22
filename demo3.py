#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import numpy
import os.path
import subprocess
import deepmodels
import json
import argparse
import time
import imageutils
import utils

def make_manifolds_knn(X,S,T,KNN,K):
  '''
  X is a string
  S is a sequence of strings
  T is a sequence of strings
  KNN is a sequence of strings
  K is a number
  '''
  S=set(S)
  T=set(T)
  P=[x for x in KNN if x!=X and x in S]
  Q=[x for x in KNN if x!=X and x in T]
  return P[:K],Q[:K]

if __name__=='__main__':
  # configure by command-line arguments
  parser=argparse.ArgumentParser(description='Generate inpainting transformations.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--backend',type=str,default='caffe+scipy',choices=['torch','caffe+scipy'],help='reconstruction implementation')
  parser.add_argument('--device_id',type=int,default=0,help='zero-indexed CUDA device')
  parser.add_argument('--K',type=int,default=100,help='number of nearest neighbors')
  parser.add_argument('--scaling',type=str,default='beta',choices=['none','beta'],help='type of step scaling')
  parser.add_argument('--iter',type=int,default=500,help='number of reconstruction iterations')
  parser.add_argument('--postprocess',type=str,default='',help='comma-separated list of postprocessing operations')
  parser.add_argument('--delta',type=str,default='2.8',help='comma-separated list of interpolation steps')
  config=parser.parse_args()
  postprocess=set(config.postprocess.split(','))
  print(json.dumps(config.__dict__))

  # load CUDA model
  minimum_resolution=200
  if config.backend=='torch':
    import deepmodels_torch
    model=deepmodels_torch.vgg19g_torch(device_id=config.device_id)
  elif config.backend=='caffe+scipy':
    model=deepmodels.vgg19g(device_id=config.device_id)
  else:
    raise ValueError('Unknown backend')

  # download precomputed features (if needed)
  if not os.path.exists('images/utzap/Shoes/Boat Shoes/ALDO/8045627.325.jpg'):
    print('Acquire the UT-Zappos50K dataset and place the files so that images/utzap/Shoes/Boat Shoes/ALDO/8045627.325.jpg is a valid path.')
    sys.exit(1)
  if not os.path.exists('datasets/utzap/vggpool5.200x200.npz'):
    url='https://www.dropbox.com/s/mms9jcznmavsxcr/utzap_knn.npz?dl=1'
    subprocess.check_call(['wget',url,'-O','datasets/utzap/vggpool5.200x200.npz'])

  # read test data
  data=numpy.load('datasets/utzap/utzap_test_set.npz')
  X=list(data['X']) # test groups
  T=list(data['T']) # non-test groups
  data=numpy.load('datasets/utzap/vggpool5.200x200.npz')
  KNN_A=data['A'] # precomputed nearest neighbors for masked X \cup T
  filelist=list(data['filelist'])
  assert filelist==[x.replace('utzap','utzap_inpaint').replace('.jpg','.png') for x in X+T]
  filelist=X+T

  # comment out the line below to generate all the test images
  X=X[:1]

  # Set the free parameters
  K=100
  delta_params=[float(x.strip()) for x in config.delta.split(',')]

  t0=time.time()
  result=[]
  original=[]
  # for each test image
  for i in range(len(X)):
    result.append([])
    xX=X[i]
    o=imageutils.read(xX)
    image_dims=(200,200)
    o=imageutils.resize(o,image_dims)
    utils.center_mask_inplace(o)
    XF=model.mean_F([o])
    original.append(o)
    KNN=[filelist[j] for j in KNN_A[filelist.index(X[i])]]
    P,Q=make_manifolds_knn(X[i],T,T,KNN,K)
    xP=P
    xQ=Q
    PF=model.mean_F(utils.image_feed_masked(xP[:K],image_dims))
    QF=model.mean_F(utils.image_feed(xQ[:K],image_dims))
    if config.scaling=='beta':
      WF=(QF-PF)/((QF-PF)**2).mean()
    elif config.scaling=='none':
      WF=(QF-PF)
    max_iter=config.iter
    init=o
    # for each interpolation step
    for delta in delta_params:
      print(xX,delta)
      t2=time.time()
      Y=model.F_inverse(XF+WF*delta,max_iter=max_iter,initial_image=init)
      t3=time.time()
      print('{} minutes to reconstruct'.format((t3-t2)/60.0))
      result[-1].append(Y)
      max_iter=config.iter//2
      init=Y
  result=numpy.asarray(result)
  original=numpy.asarray(original)
  if 'color' in postprocess:
    result=utils.color_match(numpy.expand_dims(original,1),result)
  m=imageutils.montage(numpy.concatenate([numpy.expand_dims(original,1),result],axis=1).transpose(1,0,2,3,4))
  imageutils.write('results/demo3.png',m)
  print('Output is results/demo3.png')
  t1=time.time()
  print('{} minutes ({} minutes per image).'.format((t1-t0)/60.0,(t1-t0)/60.0/result.shape[0]/result.shape[1]))

