#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import execnet

if __name__=='__channelexec__':
  import os
  os.chdir('models/facemodel')
  import numpy
  import os.path
  import sys
  import StringIO
  import traceback
  import alignface
  import facemodel
  import cPickle as pickle

  fm=facemodel.FaceModel()
  face_d,face_p=alignface.load_face_detector(predictor_path='shape_predictor_68_face_landmarks.dat')

  while True:
    args,kwargs=channel.receive()
    try:
      if len(args)<2:
        raise RuntimeError('invalid command')
      command=args[0]
      responseid=args[1]
      args=args[2:]
      if command=='ping':
        channel.send((('okay',responseid),{}))
      elif command=='forward_landmarks':
        # X is pickled N x H x W x 3 images
        # Xlm is pickled N x 68 x 2 landmarks
        # scores is pickled N x K attribute scores
        X=pickle.loads(kwargs['X'])
        Xlm=pickle.loads(kwargs['Xlm'])
        result=[]
        for x,lm in zip(X,Xlm):
          # lm,x=alignface.detect_landmarks('../../results/008994small.png',face_d,face_p)
          M,loss=alignface.fit_face_landmarks(lm,fm.celeba_template,image_dims=[160,160],twoscale=False)
          Y=alignface.warp_to_template(x,M,image_dims=[160,160])
          Y=numpy.expand_dims(Y,0)
          Y=fm.preprocess(Y)
          Y=fm.forward(Y)
          scores=fm.predict_attributes(Y)
          result.append(scores[0])
        result=numpy.asarray(result)
        channel.send((('okay',responseid),{'scores':pickle.dumps(result)}))
      elif command=='forward_images':
        # X is pickled list of N strings
        # scores is pickled N x K attributes scores
        X=pickle.loads(kwargs['X'])
        X=[(x if os.path.isabs(x) else os.path.join('../..',x)) for x in X]
        result=[]
        for ipath in X:
          lm,x=alignface.detect_landmarks(ipath,face_d,face_p)
          M,loss=alignface.fit_face_landmarks(lm,fm.celeba_template,image_dims=[160,160],twoscale=False)
          Y=alignface.warp_to_template(x,M,image_dims=[160,160])
          Y=numpy.expand_dims(Y,0)
          Y=fm.preprocess(Y)
          Y=fm.forward(Y)
          scores=fm.predict_attributes(Y)
          result.append(scores[0])
        result=numpy.asarray(result)
        channel.send((('okay',responseid),{'scores':pickle.dumps(result)}))
      elif command=='fail':
        raise RuntimeError('simulated failure')
      else:
        raise RuntimeError('unknown command')
    except:
      s=StringIO.StringIO()
      traceback.print_exc(file=s)
      channel.send((('fail',responseid),{'traceback':s.getvalue()}))

if __name__=='__main__':
  import numpy
  import facemodel_worker
  import cPickle as pickle
  import execnet
  import alignface

  face_d,face_p=alignface.load_face_detector(predictor_path='models/shape_predictor_68_face_landmarks.dat')
  gw=execnet.makegateway()
  channel=gw.remote_exec(facemodel_worker)
  channel.send((('ping',0),{}))
  print('gw',gw.remote_status())
  print('recv',channel.receive())
  Xlm,X=alignface.detect_landmarks('results/008994small.png',face_d,face_p)
  X=numpy.expand_dims(X,0)
  Xlm=numpy.expand_dims(Xlm,0)
  channel.send((('forward_landmarks',1),{'X':pickle.dumps(X),'Xlm':pickle.dumps(Xlm)}))
  print('gw',gw.remote_status())
  args,kwargs=channel.receive()
  print('recv',args[0],args[1],[pickle.loads(x) for x in args[2:]],['{} = {}'.format(k,pickle.loads(v)) for (k,v) in kwargs.iteritems()])
  channel.send((('forward_images',2),{'X':pickle.dumps(['results/008994small.png'])}))
  print('gw',gw.remote_status())
  args,kwargs=channel.receive()
  print('recv',args[0],args[1],[pickle.loads(x) for x in args[2:]],['{} = {}'.format(k,pickle.loads(v)) for (k,v) in kwargs.iteritems()])
  def error_handler(args,kwargs):
    if len(args)>=1 and args[0]=='fail':
      if 'traceback' in kwargs:
        raise RuntimeError('remote command failed\n\n'+kwargs['traceback'][:-1])
      else:
        raise RuntimeError('remote command failed (no traceback available)')
  channel.send((('fail',3),{}))
  print('gw',gw.remote_status())
  args,kwargs=channel.receive()
  error_handler(args,kwargs)
  print(args,kwargs)

