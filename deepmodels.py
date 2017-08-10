#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import numpy
import os
import os.path
import json
import scipy.spatial.distance
import cPickle as pickle

from interface import interface, implements, method

class DeepFeatureRep(interface):
  r'''
  mean_F: \mathbb{R}^{N x H x W x 3} \to \mathbb{R}^{D}
  F_inverse: \mathbb{R}^{D} \to \mathbb{R}^{H x W x 3}

  mean_F achieves O(1) space wrt N if X is an iterator that yields
  \mathbb{R}^{H x W x 3}.
  '''
  mean_F=method(['self','X'])
  F_inverse=method(['self','F','initial_image'],keywords='options')

class AttributeClassifier(interface):
  r'''
  lookup_scores: returns precomputed scores given a list of strings
  score: \mathbb{R}^{N x H x W x 3} \to \mathbb{R}^{N x K}
  fields: list of K strings
  select: constraints is a list of 2-tuples (field index, value),
    attributes is \mathbb{R}^{K}. Returns a list of image file paths which
    satisfy the given constraints and are sorted by minkowski distance
    (over some set of fields) to the given attribute vector.

  score also accepts a list of N strings.
  '''
  lookup_scores=method(['self','X'],keywords='options')
  score=method(['self','X'],keywords='options')
  fields=method(['self'])
  select=method(['self','constraints','attributes'],keywords='options')

class FaceAttributeClassifier(interface):
  r'''
  lookup_scores: returns precomputed scores given a list of strings
  lookup_landmarks: returns precomputed landmarks given a list of strings
  score: \mathbb{R}^{N x H x W x 3} \to \mathbb{R}^{N x K}
  fields: list of K strings
  select: constraints is a list of 2-tuples (field index, value),
    attributes is \mathbb{R}^{K}. Returns a list of image file paths which
    satisfy the given constraints and are sorted by minkowski distance
    (over some set of fields) to the given attribute vector.

  score also accepts a list of N strings.
  '''
  lookup_scores=method(['self','X'],keywords='options')
  lookup_landmarks=method(['self','X'],keywords='options')
  score=method(['self','X'],keywords='options')
  fields=method(['self'])
  select=method(['self','constraints','attributes'],keywords='options')

def import_caffe(device_id):
  '''
  Imports caffe with logging suppressed (unless the environment defines
  GLOG_minloglevel).
  '''
  if 'GLOG_minloglevel' not in os.environ:
    os.environ['GLOG_minloglevel']='2'
    import caffe
    del os.environ['GLOG_minloglevel']
  else:
    import caffe
  caffe.set_mode_gpu()
  caffe.set_device(device_id)
  return caffe

def caffe_unflatten(F,net,caffe_in,blob_names):
  '''
  F is \mathbb{R}^{D}
  caffe_in is \mathbb{R}^{N x 3 x H x W}, BGR

  Computes the reverse of F=numpy.concatenate([net.blobs[k].data.copy().ravel() for k in blob_names])

  Returns {k:blob[k] for k in blob_names}
  '''
  input_blob=net.inputs[0]
  net.blobs[input_blob].reshape(*caffe_in.shape)
  net.reshape()
  net.forward_all(**{input_blob:caffe_in})
  F2={}
  index=0
  for k in blob_names:
    shape=net.blobs[k].shape
    n=numpy.prod(shape)
    F2[k]=F[index:index+n].reshape(*shape)
    index+=n
  if index!=len(F):
    raise ValueError('Length of F does not match blob shapes.')
  return F2

import scipy.optimize
import deepart
import time
def dfi_reconstruct(F2,net,blob_names,caffe_in,max_iter=2000,verbose=False,tv_lambda=0.001,tv_beta=2,bounds=(-128,162)):
  '''
  F2 is a dictionary of blobs
  caffe_in is \mathbb{R}^{N x 3 x H x W}, BGR
  '''
  t0=time.time()
  blob_weights=[1.0]*len(blob_names)
  all_target_blob_names=list(blob_names)
  targets=[]
  target_data_list=[]
  for k,v in zip(blob_names,blob_weights):
    if len(targets)>0 and targets[-1][3]==v:
      targets[-1][1].append(k)
      target_data_list[-1][k]=F2[k]
    else:
      targets.append((None,[k],False,v))
      target_data_list.append({k: F2[k]})

  in_=net.inputs[0]
  if tuple(net.blobs[in_].data.shape)!=tuple(caffe_in.shape):
    net.blobs[in_].reshape(*caffe_in.shape)
    net.reshape()
  net.blobs[in_].data[...]=caffe_in

  x0=caffe_in.ravel().astype(numpy.float64)
  solver_type='L-BFGS-B'
  solver_param={'maxiter': max_iter, 'iprint': -1}
  #solver_param={'maxiter': max_iter, 'iprint': 1}
  loss0,_=deepart.objective_func(x0,net,all_target_blob_names,targets,target_data_list,tv_lambda,tv_beta)
  opt_res=scipy.optimize.minimize(deepart.objective_func,x0,args=(net,all_target_blob_names,targets,target_data_list,tv_lambda,tv_beta),bounds=zip(numpy.full_like(x0,bounds[0]),numpy.full_like(x0,bounds[1])),method=solver_type,jac=True,options=solver_param)
  loss1,_=deepart.objective_func(opt_res.x,net,all_target_blob_names,targets,target_data_list,tv_lambda,tv_beta)

  data=opt_res.x.reshape(*caffe_in.shape)
  if verbose:
    #print(opt_res)
    print('reconstruct finished in {} minutes. loss: {} -> {}'.format((time.time()-t0)/60.0,loss0,loss1))
  return data

def caffe_mean_F(X,net,input_blob,blob_names,mean,scale):
  X=iter(X)
  n=0
  for x in X:
    caffe_in=((x[numpy.newaxis,:,:,::-1]*255-mean)*scale).transpose(0,3,1,2)
    if n==0:
      net.blobs[input_blob].reshape(*caffe_in.shape)
      net.reshape()
      net.forward_all(**{input_blob:caffe_in})
    else:
      net.forward_all(**{input_blob:caffe_in})
    if n==0:
      mu={k:net.blobs[k].data.copy() for k in blob_names}
    else:
      for k in blob_names:
        mu[k]+=net.blobs[k].data.copy()
    n+=1
  if n>1:
    for k in blob_names:
      mu[k]/=float(n)
  return mu

def caffe_mean_F_flatten(X,net,input_blob,blob_names,mean,scale):
  mu=caffe_mean_F(X,net,input_blob,blob_names,mean,scale)
  return numpy.concatenate([mu[k].ravel() for k in blob_names])

def caffe_hypercolumn(F,blob_names):
  k0=blob_names[0]
  shape=(F[k0].shape[0],sum(F[k].shape[1] for k in blob_names),F[k0].shape[2],F[k0].shape[3])
  result=[F[k0]]
  for k in blob_names[1:]:
    zoom=(1.0,1.0,float(F[k0].shape[2])/F[k].shape[2],float(F[k0].shape[3])/F[k].shape[3])
    result.append(scipy.ndimage.interpolation.zoom(F[k],zoom,order=1,mode='nearest'))
  return numpy.concatenate(result,axis=1)

@implements(DeepFeatureRep)
class vgg19g(object):
  '''
  >>> model=vgg19g(device_id=0)
  >>> original=skimage.io.imread('tests/008994small.png')/255.0
  >>> ref=numpy.load('tests/008994small_reference_F.npy')
  >>> F=model.mean_F([original])
  >>> F.shape
  (4105728,)
  >>> numpy.allclose(F,ref,1e-5,1e-4)
  True
  >>> result=model.F_inverse(F,original,max_iter=5,tv_lambda=0)
  >>> result.shape
  (432, 363, 3)
  >>> numpy.allclose(original,result,1e-5,1e-4)
  True
  '''
  def __init__(self,**options):
    self.device_id=options.get('device_id',0)
    self.deploy='models/VGG_CNN_19/VGG_ILSVRC_19_layers_deploy_fullconv.prototxt'
    self.weights='models/VGG_CNN_19/vgg_normalised.caffemodel'
    self.caffe=import_caffe(self.device_id)
    self.net=self.caffe.Net(self.deploy,self.weights,self.caffe.TEST)
    self.mean=numpy.asarray((103.939, 116.779, 123.68))
    self.blob_names=['conv3_1','conv4_1','conv5_1']
    self.input_blob=self.net.inputs[0]
    self.tv_lambda=0.001
    self.max_iter=2000
  def mean_F(self,X):
    return caffe_mean_F_flatten(X,self.net,self.input_blob,self.blob_names,self.mean,1)
  def F_inverse(self,F,initial_image,**options):
    tv_lambda=options.get('tv_lambda',self.tv_lambda)
    max_iter=options.get('max_iter',self.max_iter)
    caffe_in=numpy.expand_dims((initial_image[:,:,::-1]*255-self.mean).transpose(2,0,1),0)
    F2=caffe_unflatten(F,self.net,caffe_in,self.blob_names)
    data=dfi_reconstruct(F2,self.net,self.blob_names,caffe_in,tv_lambda=tv_lambda,max_iter=max_iter)
    return ((data[0].transpose(1,2,0)+self.mean)[:,:,::-1]/255.0).clip(0,1)

@implements(AttributeClassifier)
class lfw_attributes(object):
  '''
  >>> classifier=lfw_attributes()
  >>> len(classifier.fields())
  73
  >>> classifier.fields()[0]
  'Male'
  >>> numpy.allclose(classifier.score(['Aaron_Eckhart/Aaron_Eckhart_0001.jpg'])[0,0],1.56834639173)
  True
  '''
  def __init__(self,**options):
    data=numpy.load('datasets/lfw/lfw_attributes.npz')
    self._fields=tuple(data['fields'])
    self.filelist=tuple(data['filelist'])
    self._score=data['score']
    self.map_name={k:i for i,k in enumerate(self.filelist)}
    self.rev_map_name={i:k for i,k in enumerate(self.filelist)}
  def lookup_scores(self,X,**options):
    raise NotImplementedError
  def score(self,X,**options):
    idx=[self.map_name[x] for x in X]
    return self._score[idx]
  def fields(self):
    return self._fields
  def select(self,constraints,attributes,**options):
    raise NotImplementedError

@implements(AttributeClassifier)
class celeba_attributes(object):
  def __init__(self,**options):
    data=numpy.load('datasets/celeba/list_attr_celeba.npz')
    self._fields=tuple(data['fields'])
    self.filelist=tuple(data['filelist'])
    self._score=data['score']
    self._binary_score=(self._score>=0)
    self._confident=numpy.ones_like(self._score,dtype=numpy.bool)
    self.map_name={k:i for i,k in enumerate(self.filelist)}
    self.rev_map_name={i:k for i,k in enumerate(self.filelist)}
    self.distance_idx=numpy.arange(len(self._fields))
  def lookup_scores(self,X,**options):
    raise NotImplementedError
  def score(self,X,**options):
    idx=[self.map_name[x] for x in X]
    return self._score[idx]
  def fields(self):
    return self._fields
  def select(self,constraints,attributes,**options):
    binary_attributes=(attributes>=0)
    def admissible(i):
      return all(self._binary_score[i,j]==v and self._confident[i,j] for j,v in constraints)
    S=numpy.asarray([i for i in range(len(self.filelist)) if admissible(i)])
    knn=numpy.argsort(scipy.spatial.distance.cdist(numpy.expand_dims(attributes[self.distance_idx],0),self._binary_score[S][:,self.distance_idx],'minkowski',1)).astype(numpy.int32)
    return [self.filelist[i] for i in S[knn[0]]]

@implements(FaceAttributeClassifier)
class facemodel_attributes(object):
  '''
  >>> classifier=facemodel_attributes()
  >>> len(classifier.fields())
  71
  >>> classifier.fields()[0]
  '5_o_Clock_Shadow'
  >>> classifier.lookup_scores(['images/facemodel/celeba/000009.jpg'])[0,:5]
  array([-2.41198874,  0.58286995,  0.94109076, -0.70556468, -2.84208179])
  >>> numpy.allclose(classifier.score(['tests/000009.jpg'])[0,:5],[-2.39323288,  0.5700731 ,  0.917334  , -0.66668013, -2.66346439])
  True
  '''
  def __init__(self,**options):
    data=numpy.load('datasets/facemodel/attributes.npz')
    self._fields=( '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young', 'asian', 'baby', 'black', 'brown_eyes', 'child', 'color_photo', 'eyes_open', 'flash', 'flushed_face', 'frowning', 'fully_visible_forehead', 'harsh_lighting', 'indian', 'middle_aged', 'mouth_closed', 'mouth_wide_open', 'no_eyewear', 'obstructed_forehead', 'outdoor', 'partially_visible_forehead', 'posed_photo', 'round_face', 'round_jaw', 'senior', 'shiny_skin', 'soft_lighting', 'square_face', 'strong_nose_mouth_lines', 'sunglasses', 'teeth_not_visible', 'white' )
    self.filelist=tuple(data['filelist'])
    self._scores=data['scores']
    self._landmarks=data['landmarks']
    self._binary_score=(self._scores>=0)
    # mark 70% as confident
    self._confident=numpy.zeros_like(self._scores,dtype=numpy.bool)
    self._confident[self._scores>numpy.percentile(self._scores[self._scores>=0],30,axis=0)]=True
    self._confident[self._scores<numpy.percentile(self._scores[self._scores<0],70,axis=0)]=True
    self.map_name={k:i for i,k in enumerate(self.filelist)}
    self.rev_map_name={i:k for i,k in enumerate(self.filelist)}
    # only attributes related to faces for nearest neighbor distance
    self.distance_idx=[self._fields.index(x) for x in [ '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Lipstick', 'Young', 'asian', 'baby', 'black', 'brown_eyes', 'child', 'eyes_open', 'frowning', 'fully_visible_forehead', 'indian', 'middle_aged', 'mouth_closed', 'mouth_wide_open', 'no_eyewear', 'obstructed_forehead', 'partially_visible_forehead', 'senior', 'shiny_skin', 'strong_nose_mouth_lines', 'sunglasses', 'teeth_not_visible', 'white' ]]
    # start classifer in another process since Torch and Caffe cannot
    # coexist in the same process
    import facemodel_worker
    import execnet
    self.group=execnet.Group()
    self.gw=self.group.makegateway()
    self.channel=self.gw.remote_exec(facemodel_worker)
    self.responseid=0
    self.channel.send((('ping',self.responseid),{}))
    args,kwargs=self.channel.receive()
    self.raise_remote_errors(args,kwargs)
    assert args[0]=='okay'
    assert args[1]==self.responseid
    self.responseid+=1
  def raise_remote_errors(self,args,kwargs):
    if len(args)>=1 and args[0]=='fail':
      if 'traceback' in kwargs:
        raise RuntimeError('remote command failed\n\n'+kwargs['traceback'][:-1])
      else:
        raise RuntimeError('remote command failed (no traceback available)')
  def __del__(self):
    self.group.terminate(timeout=1.0)
  def lookup_scores(self,X,**options):
    idx=[self.map_name[x] for x in X]
    return self._scores[idx]
  def lookup_landmarks(self,X,**options):
    idx=[self.map_name[x] for x in X]
    return self._landmarks[idx]
  def score(self,X,**options):
    if 'landmarks' in options:
      self.channel.send((('forward_landmarks',self.responseid),{'X':pickle.dumps(X),'Xlm':pickle.dumps(landmarks)}))
    else:
      self.channel.send((('forward_images',self.responseid),{'X':pickle.dumps(X)}))
    args,kwargs=self.channel.receive()
    self.raise_remote_errors(args,kwargs)
    assert args[0]=='okay'
    assert args[1]==self.responseid
    self.responseid+=1
    return pickle.loads(kwargs['scores'])
  def fields(self):
    return self._fields
  def select(self,constraints,attributes,**options):
    binary_attributes=(attributes>=0)
    def admissible(i):
      return all(self._binary_score[i,j]==v and self._confident[i,j] for j,v in constraints)
    S=numpy.asarray([i for i in range(len(self.filelist)) if admissible(i)])
    if len(S)<1: return []
    knn=numpy.argsort(scipy.spatial.distance.cdist(numpy.expand_dims(attributes[self.distance_idx],0),self._binary_score[S][:,self.distance_idx],'minkowski',1)).astype(numpy.int32)
    return [self.filelist[i] for i in S[knn[0]]]

if __name__=='__main__':
  import doctest
  import sys
  import skimage.io
  nfail,ntest=doctest.testmod(verbose=False)
  if nfail>0: sys.exit(1)
  print('Success.')

