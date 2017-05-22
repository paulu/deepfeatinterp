#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import time
timestamp=int(round(time.time()))
import numpy
import deepmodels
import json
import os.path
import argparse
import alignface
import imageutils
import utils

def fit_submanifold_landmarks_to_image(template,original,Xlm,face_d,face_p,landmarks=list(range(68))):
  '''
  Fit the submanifold to the template and take the top-K.

  Xlm is a N x 68 x 2 list of landmarks.
  '''
  lossX=numpy.empty((len(Xlm),),dtype=numpy.float64)
  MX=numpy.empty((len(Xlm),2,3),dtype=numpy.float64)
  nfail=0
  for i in range(len(Xlm)):
    lm=Xlm[i]
    try:
      M,loss=alignface.fit_face_landmarks(Xlm[i],template,landmarks=landmarks,image_dims=original.shape[:2])
      lossX[i]=loss
      MX[i]=M
    except alignface.FitError:
      lossX[i]=float('inf')
      MX[i]=0
      nfail+=1
  if nfail>1:
    print('fit submanifold, {} errors.'.format(nfail))
  a=numpy.argsort(lossX)
  return a,lossX,MX

if __name__=='__main__':
  # configure by command-line arguments
  parser=argparse.ArgumentParser(description='Generate high resolution face transformations.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('method',type=str,choices=['older','younger','facehair'],help='desired transformation')
  parser.add_argument('input',type=str,nargs='+',help='input color image')
  parser.add_argument('--backend',type=str,default='torch',choices=['torch','caffe+scipy'],help='reconstruction implementation')
  parser.add_argument('--device_id',type=int,default=0,help='zero-indexed CUDA device')
  parser.add_argument('--K',type=int,default=100,help='number of nearest neighbors')
  parser.add_argument('--scaling',type=str,default='none',choices=['none','beta'],help='type of step scaling')
  parser.add_argument('--iter',type=int,default=500,help='number of reconstruction iterations')
  parser.add_argument('--postprocess',type=str,default='mask',help='comma-separated list of postprocessing operations')
  parser.add_argument('--delta',type=str,default='3.5',help='comma-separated list of interpolation steps')
  parser.add_argument('--output_format',type=str,default='png',choices=['png','jpg'],help='output image format')
  parser.add_argument('--comment',type=str,default='',help='the comment is appended to the output filename')
  config=parser.parse_args()
  postprocess=set(config.postprocess.split(','))
  print(json.dumps(config.__dict__))

  # load models
  minimum_resolution=200
  if config.backend=='torch':
    import deepmodels_torch
    model=deepmodels_torch.vgg19g_torch(device_id=config.device_id)
  elif config.backend=='caffe+scipy':
    model=deepmodels.vgg19g(device_id=config.device_id)
  else:
    raise ValueError('Unknown backend')
  classifier=deepmodels.facemodel_attributes()
  fields=classifier.fields()
  gender=fields.index('Male')
  smile=fields.index('Smiling')
  face_d,face_p=alignface.load_face_detector()

  # Set the free parameters
  K=config.K
  delta_params=[float(x.strip()) for x in config.delta.split(',')]

  X=config.input

  t0=time.time()
  opathlist=[]
  # for each test image
  for i in range(len(X)):
    xX=X[i]
    template,original=alignface.detect_landmarks(xX,face_d,face_p)
    image_dims=original.shape[:2]
    if min(image_dims)<minimum_resolution:
      s=float(minimum_resolution)/min(image_dims)
      image_dims=(int(round(image_dims[0]*s)),int(round(image_dims[1]*s)))
      original=imageutils.resize(original,image_dims)
    XF=model.mean_F([original])
    XA=classifier.score([xX])[0]
    print(xX,', '.join(k for i,k in enumerate(fields) if XA[i]>=0))

    # select positive and negative sets based on gender and mouth
    if config.method=='older':
      cP=[(gender,XA[gender]>=0),(smile,XA[smile]>=0),(fields.index('Young'),True)]
      cQ=[(gender,XA[gender]>=0),(smile,XA[smile]>=0),(fields.index('Young'),False)]
    elif config.method=='younger':
      cP=[(gender,XA[gender]>=0),(smile,XA[smile]>=0),(fields.index('Young'),False)]
      cQ=[(gender,XA[gender]>=0),(smile,XA[smile]>=0),(fields.index('Young'),True)]
    elif config.method=='facehair':
      cP=[(gender,XA[gender]>=0),(smile,XA[smile]>=0),(fields.index('No_Beard'),True),(fields.index('Mustache'),False)]
      cQ=[(gender,XA[gender]>=0),(smile,XA[smile]>=0),(fields.index('No_Beard'),False),(fields.index('Mustache'),True)]
    else:
      raise ValueError('Unknown method')
    P=classifier.select(cP,XA)
    Q=classifier.select(cQ,XA)
    if len(P)<4*K or len(Q)<4*K:
      print('{}: Not enough images in database (|P|={}, |Q|={}).'.format(xX,len(P),len(Q)))
      continue

    # fit the best 4K database images to input image
    Plm=classifier.lookup_landmarks(P[:4*K])
    Qlm=classifier.lookup_landmarks(Q[:4*K])
    idxP,lossP,MP=fit_submanifold_landmarks_to_image(template,original,Plm,face_d,face_p)
    idxQ,lossQ,MQ=fit_submanifold_landmarks_to_image(template,original,Qlm,face_d,face_p)

    # Use the K best fitted images
    xP=[P[i] for i in idxP[:K]]
    xQ=[Q[i] for i in idxQ[:K]]
    PF=model.mean_F(utils.warped_image_feed(xP,MP[idxP[:K]],image_dims))
    QF=model.mean_F(utils.warped_image_feed(xQ,MQ[idxQ[:K]],image_dims))
    if config.scaling=='beta':
      WF=(QF-PF)/((QF-PF)**2).mean()
    elif config.scaling=='none':
      WF=(QF-PF)
    max_iter=config.iter
    init=original

    # for each interpolation step
    result=[]
    for delta in delta_params:
      print(xX,image_dims,delta,len(xP),len(xQ))
      t2=time.time()
      Y=model.F_inverse(XF+WF*delta,max_iter=max_iter,initial_image=init)
      t3=time.time()
      print('{} minutes to reconstruct'.format((t3-t2)/60.0))
      result.append(Y)
      max_iter=config.iter//2
      init=Y
    result=numpy.asarray([result])
    original=numpy.asarray([original])
    prefix_path=os.path.splitext(xX)[0]
    X_mask=prefix_path+'-mask.png'
    if 'mask' in postprocess and os.path.exists(X_mask):
      mask=imageutils.resize(imageutils.read(X_mask),image_dims)
      result*=mask
      result+=original*(1-mask)
    if 'color' in postprocess:
      result=utils.color_match(numpy.asarray([original]),result)
    if 'mask' in postprocess and os.path.exists(X_mask):
      result*=mask
      result+=original*(1-mask)
    m=imageutils.montage(numpy.concatenate([numpy.expand_dims(original,1),result],axis=1))
    opath='{}_{}_{}{}.{}'.format(prefix_path,timestamp,config.method,'_'+config.comment if config.comment else '',config.output_format)
    imageutils.write(opath,m)
    opathlist.append(opath)
  print('Outputs are {}'.format(' '.join(opathlist)))
  t1=time.time()
  print('{} minutes ({} minutes per image).'.format((t1-t0)/60.0,(t1-t0)/60.0/len(X)/len(delta_params)))

