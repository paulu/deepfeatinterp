#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import time
import numpy
import deepmodels
import json
import os.path
import argparse
import subprocess
import pipes
import alignface
import imageutils
import utils

if __name__=='__main__':
  # configure by command-line arguments
  parser=argparse.ArgumentParser(description='Generate high resolution face transformations.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('input',type=str,help='input color image')
  parser.add_argument('data',type=str,help='input data file (produced by --extradata)')
  parser.add_argument('start_delta',type=float,help='initial interpolation step')
  parser.add_argument('stop_delta',type=float,help='maximum interpolation step')
  parser.add_argument('delta_step',type=float,help='change in interpolation step per frame')
  parser.add_argument('step_iter',type=int,help='number of reconstruction iterations for subsequent frames')
  parser.add_argument('--backend',type=str,default='torch',choices=['torch','caffe+scipy'],help='reconstruction implementation')
  parser.add_argument('--device_id',type=int,default=0,help='zero-indexed CUDA device')
  parser.add_argument('--iter',type=int,default=500,help='number of reconstruction iterations for the first frame')
  parser.add_argument('--postprocess',type=str,default='mask',help='comma-separated list of postprocessing operations')
  parser.add_argument('--output_format',type=str,default='png',choices=['png','jpg'],help='output image format')
  parser.add_argument('--comment',type=str,default='',help='the comment is appended to the output filename')
  config=parser.parse_args()
  postprocess=set(config.postprocess.split(','))
  postfix_comment='_'+config.comment if config.comment else ''
  print(json.dumps(config.__dict__))

  # args: original datafile initial_delta stepsize startframe endframe device_id1 device_id2
  ipath1=config.input
  assert os.path.exists(ipath1)
  ipath2=config.data
  assert os.path.exists(ipath2)
  delta0=config.start_delta
  delta1=config.stop_delta
  assert 0<=delta0
  assert delta0<=delta1
  ddelta=config.delta_step
  assert ddelta>0

  # load models
  if config.backend=='torch':
    import deepmodels_torch
    model=deepmodels_torch.vgg19g_torch(device_id=config.device_id)
  elif config.backend=='caffe+scipy':
    model=deepmodels.vgg19g(device_id=config.device_id)
  else:
    raise ValueError('Unknown backend')

  # load interpolation data
  original=imageutils.read(ipath1)
  XF=model.mean_F([original])
  prefix_path=os.path.splitext(ipath1)[0]
  if not os.path.exists(prefix_path+postfix_comment): os.mkdir(prefix_path+postfix_comment) # debug
  X_mask=prefix_path+'-mask.png'
  if 'mask' in postprocess and os.path.exists(X_mask):
    mask=imageutils.resize(imageutils.read(X_mask),image_dims)
  data=numpy.load(ipath2)
  if 'WF' in data:
    WF=data['WF']
  elif 'muQ' in data and 'muP' in data:
    WF=data['muQ']-data['muP']

  # generate frames
  t0=time.time()
  max_iter=config.iter
  init=original
  i=0
  delta=delta0
  while delta<=delta1:
    print('Starting frame #{:06}, {}'.format(i+1,(time.time()-t0)/60.0))
    result=model.F_inverse(XF+WF*delta,max_iter=max_iter,initial_image=init)
    max_iter=config.step_iter
    init=result
    i=i+1
    delta=delta+ddelta
    if 'mask' in postprocess and os.path.exists(X_mask):
      result*=mask
      result+=original*(1-mask)
    if 'color' in postprocess:
      result=utils.color_match(numpy.asarray([original]),numpy.asarray([[result]]))[0,0]
    if 'mask' in postprocess and os.path.exists(X_mask):
      result*=mask
      result+=original*(1-mask)
    imageutils.write(prefix_path+postfix_comment+'/{:06}.png'.format(i),result)

  # generate movie
  cmd=['ffmpeg','-y','-f','image2','-i',prefix_path+postfix_comment+'/%06d.png','-vcodec','libx264','-crf','19','-g','60','-r','30','-s','{}x{}'.format(original.shape[1],original.shape[0]),prefix_path+postfix_comment+'_movie.mkv']
  print(' '.join(pipes.quote(x) for x in cmd))
  subprocess.check_call(cmd)
  print('Output is {}'.format(prefix_path+postfix_comment+'_movie.mkv'))
