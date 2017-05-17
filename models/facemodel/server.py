#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import numpy
import sys
import pickle
import facemodel
import alignface
import traceback

if __name__=='__main__':
  '''
  This is a simple server process. Pass an escaped pickled pathname to its stdin
  and it will return an escaped pickled numpy array of attribute scores. If it
  returns None then there was an error.

  Run with "python -W ignore -u" for suppressed warnings and unbuffered stdout.
  '''
  fm=facemodel.FaceModel()
  face_d,face_p=alignface.load_face_detector(predictor_path='shape_predictor_68_face_landmarks.dat')
  print('facemodel','ready',file=sys.stderr)
  while True:
    rawdata=sys.stdin.readline()
    try:
      ipath=pickle.loads(rawdata.decode('string_escape'))
      #print('facemodel','got',ipath,file=sys.stderr)
      if ipath is None: break
      M,X,revmask,original,loss=alignface.fit_face(ipath,face_d,face_p,fm.celeba_template,image_dims=[160,160],twoscale=False)
      X=numpy.expand_dims(X,0)
      X=fm.preprocess(X)
      #print('facemodel','X',X.shape,X.min(),X.mean(),X.max(),file=sys.stderr)
      Y=fm.forward(X)
      scores=fm.predict_attributes(Y)
      print(pickle.dumps(scores).encode('string_escape'),file=sys.stdout)
      sys.stdout.flush()
    except:
      traceback.print_exc(file=sys.stderr)
      print(pickle.dumps(None).encode('string_escape'),file=sys.stdout)
      sys.stdout.flush()
  print('facemodel','stop',file=sys.stderr)
