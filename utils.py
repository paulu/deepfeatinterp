
from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import numpy
import imageutils
import alignface

def image_feed(S,image_dims):
  '''
  Given a list of file paths and a 2-tuple of (H, W), yields H x W x 3 images.
  '''
  for x in S:
    I=imageutils.read(x)
    if I.shape[:2]!=image_dims:
      yield imageutils.resize(I,tuple(image_dims))
    else:
      yield I

def warped_image_feed(S,MP,image_dims):
  '''
  Given a list of file paths, warp matrices and a 2-tuple of (H, W),
  yields H x W x 3 images.
  '''
  for i,x in enumerate(S):
    I=imageutils.read(x)
    yield numpy.asarray(alignface.warp_to_template(I,MP[i],image_dims=image_dims))

def center_mask_inplace(I):
  I[50:150,50:150]=0.5
  return I

def image_feed_masked(S,image_dims):
  '''
  Given a list of file paths and a 2-tuple of (H, W), yields H x W x 3 images.
  '''
  for x in S:
    I=imageutils.read(x)
    if I.shape[:2]!=image_dims:
      I=imageutils.resize(I,tuple(image_dims))
    center_mask_inplace(I)
    yield I

def color_match(A,B):
  '''
  A is a rank 5 tensor (column of original images)
  B is a rank 5 tensor (grid of images)
  '''
  A=numpy.asarray(A)
  B=numpy.asarray(B)
  print('Computing color match',A.shape,B.shape)
  m=A.reshape(A.shape[0],1,-1).mean(axis=2)
  m=numpy.expand_dims(numpy.expand_dims(numpy.expand_dims(m,-1),-1),-1)
  s=(A-m).reshape(A.shape[0],1,-1).std(axis=2)
  s=numpy.expand_dims(numpy.expand_dims(numpy.expand_dims(s,-1),-1),-1)
  m2=B.reshape(B.shape[0],B.shape[1],-1).mean(axis=2)
  m2=numpy.expand_dims(numpy.expand_dims(numpy.expand_dims(m2,-1),-1),-1)
  s2=(B-m2).reshape(B.shape[0],B.shape[1],-1).std(axis=2)
  s2=numpy.expand_dims(numpy.expand_dims(numpy.expand_dims(s2,-1),-1),-1)
  return (B-m2)*(s+1e-8)/(s2+1e-8)+m

