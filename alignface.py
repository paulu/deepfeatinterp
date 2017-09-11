#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import numpy
import numpy.linalg
import sys
import os
import glob
import skimage.io
import skimage.transform
import dlib
import math
import scipy.optimize
import cv2

class FitError(Exception):
  pass

if True:
  import PIL.Image
  import PIL.ImageFont
  import PIL.ImageDraw
  import subprocess
  def render_label(s,size,font=PIL.ImageFont.truetype(subprocess.check_output(['fc-match','-f','%{file}','Droid']),80)):
    image=PIL.Image.new('RGBA',(5,5),(255,255,255,0))
    draw=PIL.ImageDraw.Draw(image)
    image=PIL.Image.new('RGBA',draw.textsize(s,font=font),(255,255,255,0))
    draw=PIL.ImageDraw.Draw(image)
    draw.text((0,0),s,(255,255,255,255),font=font)
    if size[0]==None:
      size=list(size)
      size[0]=int(round(image.size[1]*size[1]/float(image.size[0])))
    elif size[1]==None:
      size=list(size)
      size[1]=int(round(image.size[0]*size[0]/float(image.size[1])))
    image=image.resize((size[1],size[0]),PIL.Image.LANCZOS)
    return numpy.array(image)/255.0

def warp_to_template(original,M,border_value=(0.5,0.5,0.5),image_dims=(400,400)):
  return cv2.warpAffine(original.transpose(1,0,2),M[::-1],dsize=(image_dims[1],image_dims[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=border_value)

def warp_from_template(original,M,border_value=(0.5,0.5,0.5),image_dims=(400,400)):
  return cv2.warpAffine(original,M[::-1],dsize=(image_dims[0],image_dims[1]),flags=(cv2.INTER_AREA | cv2.WARP_INVERSE_MAP),borderMode=cv2.BORDER_CONSTANT,borderValue=border_value).transpose(1,0,2)

def argmin(S,F):
  return min(((i,F(i)) for i in S),key=lambda x: x[1])[0]

def fit_face_landmarks(X,template,verbose=False,landmarks=[33,39,42,8],scale_landmarks=[39,42],location_landmark=33,image_dims=(400,400),twoscale=True):
  '''
  X is a N x 2 matrix of landmark coordinates in the frame of the original image
  template is a N x 2 matrix of landmark coordinates in the frame of the template
  image_dims is the (H,W) of the template
  '''
  Xsl=X[scale_landmarks].T.astype(numpy.float64)
  Xll=X[location_landmark].astype(numpy.float64)
  X=numpy.concatenate([X[landmarks].T,numpy.ones((1,len(landmarks)))],axis=0)

  # setup loss function
  Y=template[landmarks].T
  if twoscale:
    def f(scale1,scale2,theta,delta,X):
      ct=math.cos(theta)
      st=math.sin(theta)
      M=numpy.array([ct*scale1,-st*scale1,delta[0]*scale1,st*scale2,ct*scale2,delta[1]*scale2]).reshape(2,3)
      MXmY=(M.dot(X)-Y)
      loss=0.5*(MXmY**2).sum()
      J1=numpy.array([ct,-st,delta[0],0.0,0.0,0.0]).reshape(2,3)
      J2=numpy.array([0.0,0.0,0.0,st,ct,delta[1]]).reshape(2,3)
      J3=numpy.array([-st*scale1,-ct*scale1,0.0,ct*scale2,-st*scale2,0.0]).reshape(2,3)
      J4=numpy.array([0.0,0.0,1.0*scale1,0.0,0.0,0.0]).reshape(2,3)
      J5=numpy.array([0.0,0.0,0.0,0.0,0.0,1.0*scale2]).reshape(2,3)
      jac=numpy.array([(MXmY*(J1.dot(X))).sum(),(MXmY*(J2.dot(X))).sum(),(MXmY*(J3.dot(X))).sum(),(MXmY*(J4.dot(X))).sum(),(MXmY*(J5.dot(X))).sum()])
      return loss,jac
    def g(scale1,scale2,theta,delta):
      ct=math.cos(theta)
      st=math.sin(theta)
      M=numpy.array([ct*scale1,-st*scale1,delta[0]*scale1,st*scale2,ct*scale2,delta[1]*scale2]).reshape(2,3)
      MXmY=(M.dot(X)-Y)
      loss=0.5*(MXmY**2).sum()
      return M,loss
  else:
    def f(scale,theta,delta,X):
      ct=math.cos(theta)
      st=math.sin(theta)
      M=scale*numpy.array([ct,-st,delta[0],st,ct,delta[1]]).reshape(2,3)
      MXmY=(M.dot(X)-Y)
      loss=0.5*(MXmY**2).sum()
      J1=numpy.array([ct,-st,delta[0],st,ct,delta[1]]).reshape(2,3)
      J2=scale*numpy.array([-st,-ct,0.0,ct,-st,0.0]).reshape(2,3)
      J3=scale*numpy.array([0.0,0.0,1.0,0.0,0.0,0.0]).reshape(2,3)
      J4=scale*numpy.array([0.0,0.0,0.0,0.0,0.0,1.0]).reshape(2,3)
      jac=numpy.array([(MXmY*(J1.dot(X))).sum(),(MXmY*(J2.dot(X))).sum(),(MXmY*(J3.dot(X))).sum(),(MXmY*(J4.dot(X))).sum()])
      return loss,jac
    def g(scale,theta,delta):
      ct=math.cos(theta)
      st=math.sin(theta)
      M=scale*numpy.array([ct,-st,delta[0],st,ct,delta[1]]).reshape(2,3)
      MXmY=(M.dot(X)-Y)
      loss=0.5*(MXmY**2).sum()
      return M,loss

  # scipy optimizer
  tsl=template[scale_landmarks]
  initial_scale=min(numpy.linalg.norm(tsl[0]-tsl[1])/(numpy.linalg.norm(Xsl[:,0]-Xsl[:,1])+1e-5),max(image_dims))
  initial_delta=template[location_landmark]/initial_scale-Xll
  if twoscale:
    x0=numpy.asarray([initial_scale,initial_scale,0.0,initial_delta[0],initial_delta[1]]).astype(numpy.float64)
    def opt_fn(x0,*args):
      return f(x0[0],x0[1],x0[2],x0[3:5],*args)
    bounds=[(0,max(image_dims)),(0,max(image_dims)),(-3.1415926,3.1415926),(-(max(image_dims)**2),max(image_dims)**2),(-(max(image_dims)**2),max(image_dims)**2)]
  else:
    x0=numpy.asarray([initial_scale,0.0,initial_delta[0],initial_delta[1]]).astype(numpy.float64)
    def opt_fn(x0,*args):
      return f(x0[0],x0[1],x0[2:4],*args)
    bounds=[(0,max(image_dims)),(-3.1415926,3.1415926),(-(max(image_dims)**2),max(image_dims)**2),(-(max(image_dims)**2),max(image_dims)**2)]
  #print('check gradient')
  #print('check_grad',scipy.optimize.check_grad(lambda x0,*args: opt_fn(x0,*args)[0],lambda x0,*args: opt_fn(x0,*args)[1],x0,X))
  if verbose: print('initial guess',x0)
  result=[]
  for method in ['L-BFGS-B','TNC']:
    result.append(scipy.optimize.minimize(opt_fn,x0,args=(X,),jac=True,method=method,bounds=bounds))
  if verbose: print('{} of {} methods converged.'.format(sum(1 for x in result if x.success),len(result)))
  if not any(x.success for x in result):
    raise FitError('Cannot align face to template.\n{}'.format(result))
    for x in result: print(x)
  result=argmin(result,lambda x: x.fun)
  if verbose: print(result)
  if twoscale:
    scale1=result.x[0]
    scale2=result.x[1]
    theta=result.x[2]
    delta=result.x[3:5]
    M,loss=g(scale1,scale2,theta,delta)
  else:
    scale=result.x[0]
    theta=result.x[1]
    delta=result.x[2:4]
    M,loss=g(scale,theta,delta)
  return M,loss

def fit_face(ipath,detector,predictor,template,border_value=(0.5,0.5,0.5),upsample=0,verbose=False,landmarks=[33,39,42,8],scale_landmarks=[39,42],location_landmark=33,image_dims=(400,400),twoscale=True):
  '''
  Given an image, looks for exactly one face with DLIB then warps it to
  fit a 400x400 template. This code assumes the face is not significantly
  larger than 400x400 in the original image. If the face is small in
  the original image then set upsample to an integer greater than zero.

  ipath is a string
  detector and predictor are dlib objects
  template is a N x 2 list of landmarks

  landmarks is a list of landmark indices to fit.
  scale_landmarks is two landmark indices to initialize scale (inter-ocular landmarks work well).
  location_landmark is a landmark index to initialize position (a central landmark works well).

  Returns warp matrix, template face, original mask, original image, loss.
  '''
  original255=skimage.io.imread(ipath).astype(numpy.ubyte)
  original=original255/255.0
  dets=detector(original255,upsample)
  if len(dets)!=1: raise FitError('{}: detected zero or more than one face.'.format(ipath))

  # read detected points in original coords
  for k,d in enumerate(dets):
    shape=predictor(original255,d)
    X=numpy.array([[shape.part(i).y for i in landmarks],[shape.part(i).x for i in landmarks],[1]*len(landmarks)]).astype(numpy.float64)
    Xsl=numpy.array([[shape.part(i).y for i in scale_landmarks],[shape.part(i).x for i in scale_landmarks]]).astype(numpy.float64)
    Xll=numpy.array([shape.part(location_landmark).y,shape.part(location_landmark).x])

  # setup loss function
  Y=template[landmarks].T
  if twoscale:
    def f(scale1,scale2,theta,delta,X):
      ct=math.cos(theta)
      st=math.sin(theta)
      M=numpy.array([ct*scale1,-st*scale1,delta[0]*scale1,st*scale2,ct*scale2,delta[1]*scale2]).reshape(2,3)
      MXmY=(M.dot(X)-Y)
      loss=0.5*(MXmY**2).sum()
      J1=numpy.array([ct,-st,delta[0],0.0,0.0,0.0]).reshape(2,3)
      J2=numpy.array([0.0,0.0,0.0,st,ct,delta[1]]).reshape(2,3)
      J3=numpy.array([-st*scale1,-ct*scale1,0.0,ct*scale2,-st*scale2,0.0]).reshape(2,3)
      J4=numpy.array([0.0,0.0,1.0*scale1,0.0,0.0,0.0]).reshape(2,3)
      J5=numpy.array([0.0,0.0,0.0,0.0,0.0,1.0*scale2]).reshape(2,3)
      jac=numpy.array([(MXmY*(J1.dot(X))).sum(),(MXmY*(J2.dot(X))).sum(),(MXmY*(J3.dot(X))).sum(),(MXmY*(J4.dot(X))).sum(),(MXmY*(J5.dot(X))).sum()])
      return loss,jac
    def g(scale1,scale2,theta,delta):
      ct=math.cos(theta)
      st=math.sin(theta)
      M=numpy.array([ct*scale1,-st*scale1,delta[0]*scale1,st*scale2,ct*scale2,delta[1]*scale2]).reshape(2,3)
      MXmY=(M.dot(X)-Y)
      loss=0.5*(MXmY**2).sum()
      return M,loss
  else:
    def f(scale,theta,delta,X):
      ct=math.cos(theta)
      st=math.sin(theta)
      M=scale*numpy.array([ct,-st,delta[0],st,ct,delta[1]]).reshape(2,3)
      MXmY=(M.dot(X)-Y)
      loss=0.5*(MXmY**2).sum()
      J1=numpy.array([ct,-st,delta[0],st,ct,delta[1]]).reshape(2,3)
      J2=scale*numpy.array([-st,-ct,0.0,ct,-st,0.0]).reshape(2,3)
      J3=scale*numpy.array([0.0,0.0,1.0,0.0,0.0,0.0]).reshape(2,3)
      J4=scale*numpy.array([0.0,0.0,0.0,0.0,0.0,1.0]).reshape(2,3)
      jac=numpy.array([(MXmY*(J1.dot(X))).sum(),(MXmY*(J2.dot(X))).sum(),(MXmY*(J3.dot(X))).sum(),(MXmY*(J4.dot(X))).sum()])
      return loss,jac
    def g(scale,theta,delta):
      ct=math.cos(theta)
      st=math.sin(theta)
      M=scale*numpy.array([ct,-st,delta[0],st,ct,delta[1]]).reshape(2,3)
      MXmY=(M.dot(X)-Y)
      loss=0.5*(MXmY**2).sum()
      return M,loss

  # scipy optimizer
  tsl=template[scale_landmarks]
  initial_scale=min(numpy.linalg.norm(tsl[0]-tsl[1])/(numpy.linalg.norm(Xsl[:,0]-Xsl[:,1])+1e-5),max(image_dims))
  initial_delta=template[location_landmark]/initial_scale-Xll
  if twoscale:
    x0=numpy.asarray([initial_scale,initial_scale,0.0,initial_delta[0],initial_delta[1]]).astype(numpy.float64)
    def opt_fn(x0,*args):
      return f(x0[0],x0[1],x0[2],x0[3:5],*args)
    bounds=[(0,max(image_dims)),(0,max(image_dims)),(-3.1415926,3.1415926),(-(max(image_dims)**2),max(image_dims)**2),(-(max(image_dims)**2),max(image_dims)**2)]
  else:
    x0=numpy.asarray([initial_scale,0.0,initial_delta[0],initial_delta[1]]).astype(numpy.float64)
    def opt_fn(x0,*args):
      return f(x0[0],x0[1],x0[2:4],*args)
    bounds=[(0,max(image_dims)),(-3.1415926,3.1415926),(-(max(image_dims)**2),max(image_dims)**2),(-(max(image_dims)**2),max(image_dims)**2)]
  #print('check gradient')
  #print('check_grad',scipy.optimize.check_grad(lambda x0,*args: opt_fn(x0,*args)[0],lambda x0,*args: opt_fn(x0,*args)[1],x0,X))
  if verbose: print('initial guess',x0)
  result=[]
  for method in ['L-BFGS-B','TNC']:
    result.append(scipy.optimize.minimize(opt_fn,x0,args=(X,),jac=True,method=method,bounds=bounds))
  if verbose: print('{} of {} methods converged.'.format(sum(1 for x in result if x.success),len(result)))
  if not any(x.success for x in result):
    raise FitError('{}: cannot align face to template.\n{}'.format(ipath,result))
    for x in result: print(x)
  result=argmin(result,lambda x: x.fun)
  if verbose: print(result)
  if twoscale:
    scale1=result.x[0]
    scale2=result.x[1]
    theta=result.x[2]
    delta=result.x[3:5]
    M,loss=g(scale1,scale2,theta,delta)
  else:
    scale=result.x[0]
    theta=result.x[1]
    delta=result.x[2:4]
    M,loss=g(scale,theta,delta)
  #print(template[landmarks].T)
  #print(numpy.dot(M,X))

  # warp original image
  # cv2 upsample: cv2.INTER_LINEAR
  # cv2 downsample: cv2.INTER_AREA
  img2=warp_to_template(original,M,border_value=(0.5,0.5,0.5),image_dims=image_dims)
  revmask=warp_from_template(numpy.ones_like(img2),M,border_value=(0.0,0.0,0.0),image_dims=(original.shape[0],original.shape[1]))
  #revmask=cv2.warpAffine(numpy.ones_like(img2),M[::-1],dsize=(original.shape[0],original.shape[1]),flags=(cv2.INTER_AREA | cv2.WARP_INVERSE_MAP),borderMode=cv2.BORDER_CONSTANT,borderValue=(0.0,0.0,0.0))
  #revmask=revmask.transpose(1,0,2)
  return M,img2,revmask,original,loss

def load_face_detector(predictor_path='models/shape_predictor_68_face_landmarks.dat'):
  detector=dlib.get_frontal_face_detector()
  predictor=dlib.shape_predictor(predictor_path)
  return detector,predictor

def detect_landmarks(ipath,detector,predictor,upsample=0,image=None):
  if image is None:
    original255=skimage.io.imread(ipath).astype(numpy.ubyte)
    original=original255/255.0
  else:
    original=image
    original255=(original.clip(0,1)*255).round().astype(numpy.ubyte)
  dets=detector(original255,upsample)
  if len(dets)!=1: raise FitError('{}: detected zero or more than one face.'.format(ipath))

  for k,d in enumerate(dets):
    shape=predictor(original255,d)
    X=numpy.array([[shape.part(i).y,shape.part(i).x] for i in range(68)]).astype(numpy.float64)
  return X,original

def compute_template(globspec='images/lfw_aegan/*/*.png',image_dims=[400,400],predictor_path='models/shape_predictor_68_face_landmarks.dat',center_crop=None,subsample=1):
  # Credit: http://dlib.net/face_landmark_detection.py.html
  detector=dlib.get_frontal_face_detector()
  predictor=dlib.shape_predictor(predictor_path)

  template=numpy.zeros((68,2),dtype=numpy.float64)
  count=0

  if not center_crop is None:
    center_crop=numpy.asarray(center_crop)
    cy,cx=(numpy.asarray(image_dims)-center_crop)//2

  # compute mean landmark locations
  S=sorted(glob.glob(globspec))
  S=S[::subsample]
  for ipath in S:
    print("Processing file: {}".format(ipath))
    img=(skimage.transform.resize(skimage.io.imread(ipath)/255.0,tuple(image_dims)+(3,),order=2,mode='nearest')*255).clip(0,255).astype(numpy.ubyte)
    if not center_crop is None:
      img=img[cy:cy+center_crop[0],cx:cx+center_crop[0]]

    upsample=0
    dets=detector(img,upsample)
    if len(dets)!=1: continue

    for k,d in enumerate(dets):
      shape=predictor(img, d)
      for i in range(68):
        template[i]+=(shape.part(i).y,shape.part(i).x)
      count+=1
  template/=float(count)
  return template
  # lfw_aegan 400x400 template map
  # [[ 251.58852868  201.50275826]  # 33 where nose meets upper-lip
  #  [ 172.69409809  168.66523086]  # 39 inner-corner of left eye
  #  [ 171.72236076  232.09718129]] # 42 inner-corner or right eye

def visualize_template(opath,template,image_dims,zoom=1):
  result=numpy.zeros((image_dims[0]*zoom,image_dims[1]*zoom,3),dtype=numpy.float64)
  for j in range(len(template)):
    label=render_label(str(j),(image_dims[0]*zoom//50,None))
    py,px=int(round(template[j,0])*zoom),int(round(template[j,1])*zoom)
    dest=result[py:py+label.shape[0],px:px+label.shape[1]]
    source=label[0:dest.shape[0],0:dest.shape[1]]
    dest*=(1-source[:,:,3:4])
    dest+=source[:,:,:3]*source[:,:,3:4]
  skimage.io.imsave(opath,result)

