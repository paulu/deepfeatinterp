#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import alignface
import numpy
import os
import os.path
import sys
import PIL.Image

# In datasets/facemodel/attributes.npz:
#   scores is N x K
#   landmarks is N x 68 x 2
#   filelist is a list of files (starts with images/facemodel/...)

# In datasets/facemodel/history.npz
#   dedup_descriptors maps a filename to a descriptor vector

import npz
from facemodel_server import *

def save_image_list(opath,subdir):
  S=set(['.jpg','.png','.jpeg'])
  result=[]
  def error_fn(e): raise e
  for dirpath,dirnames,filenames in os.walk(subdir,onerror=error_fn,followlinks=True):
    for x in filenames:
      if (os.path.splitext(x)[1]).lower() in S:
        result.append(os.path.join(dirpath,x)[len(subdir)+1:])
  result.sort()
  with open(opath,'w') as f:
    for x in result:
      print(x,file=f)

def duplicate_descriptor(img):
  x=numpy.array(img.convert('L').resize((32,32),PIL.Image.BICUBIC)).ravel()
  x=x-x.mean()
  x=x/x.std()
  return (x*50/3+50).clip(0,100).astype(numpy.uint8)

def rebuild_dataset(interactive=True):
  # scans the image directory and rebuilds database
  fields = [ '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young', 'asian', 'baby', 'black', 'brown_eyes', 'child', 'color_photo', 'eyes_open', 'flash', 'flushed_face', 'frowning', 'fully_visible_forehead', 'harsh_lighting', 'indian', 'middle_aged', 'mouth_closed', 'mouth_wide_open', 'no_eyewear', 'obstructed_forehead', 'outdoor', 'partially_visible_forehead', 'posed_photo', 'round_face', 'round_jaw', 'senior', 'shiny_skin', 'soft_lighting', 'square_face', 'strong_nose_mouth_lines', 'sunglasses', 'teeth_not_visible', 'white' ]
  fields=[x.replace('_',' ') for x in fields]

  # load old attributes
  if os.path.exists('datasets/facemodel/attributes.npz'):
    data=numpy.load('datasets/facemodel/attributes.npz')
    old_scores=data['scores']
    old_landmarks=data['landmarks']
    old_filelist=data['filelist']
    old_filelist_revmap={k:i for i,k in enumerate(old_filelist)}

    # special code for old png files
    for x in old_filelist:
      if x.endswith('.png'):
        y=os.path.splitext(x)[0]+'.jpg'
        old_filelist_revmap[y]=old_filelist_revmap[x]
    print('Loaded {} old attribute entries.'.format(len(old_filelist)))
  else:
    old_filelist_revmap={}

  # scan directory for images
  # Tip: comment out the line below if you want direct control over the filelist
  save_image_list('datasets/facemodel/filelist.txt','images/facemodel')
  S=['images/facemodel/'+x.strip() for x in open('datasets/facemodel/filelist.txt').readlines()]
  print('Found {} images.'.format(len(S)))
  T=[x for x in S if x not in old_filelist_revmap]
  print('Found {} new images.'.format(len(T)))

  # load history
  if os.path.exists('datasets/facemodel/history.npz'):
    dedup_descriptors=npz.NpzLog('datasets/facemodel/history.npz',shape=(0,1024),dtype=numpy.uint8,buflen=len(T))
  else:
    dedup_descriptors=npz.NpzLog(None,shape=(0,1024),dtype=numpy.uint8,buflen=len(T))
  print('Loaded {} descriptors.'.format(len(dedup_descriptors)))

  face_d,face_p=alignface.load_face_detector()

  new_scores=[]
  new_landmarks=[]
  new_filelist=[]
  dedup=set()
  dedup_head={}
  dedup_head_size={}
  failed=set()
  dellist=set()
  count=0
  def process_dedup(ipath):
    img=PIL.Image.open(ipath)
    if ipath not in dedup_descriptors:
      descriptor=duplicate_descriptor(img)
      dedup_descriptors[ipath]=descriptor
      u=dedup_descriptors.usage()
      if ((u['n_buffer']+u['n_overflow']) % 10000)==0:
        print('dedup checkpoint',u['n_storage'],u['n_buffer'],u['n_overflow'])
        dedup_descriptors.write('datasets/facemodel/history.npz')
    else:
      descriptor=dedup_descriptors[ipath]
    descriptor=tuple(descriptor)
    if descriptor in dedup:
      # matches existing cluster
      if sum(img.size)>dedup_head_size[descriptor]:
        dellist.add(dedup_head[descriptor])
        print('dedup',ipath,dedup_head[descriptor])
        dedup_head[descriptor]=ipath
        dedup_head_size[descriptor]=sum(img.size)
      else:
        print('dedup',dedup_head[descriptor],ipath)
        dellist.add(ipath)
    else:
      # start new cluster
      dedup.add(descriptor)
      dedup_head[descriptor]=ipath
      dedup_head_size[descriptor]=sum(img.size)
  p=facemodel_server_start()
  try:
    for ipath in S:
      if ipath in old_filelist_revmap:
        # preexisting
        index=old_filelist_revmap[ipath]
        new_scores.append(old_scores[index])
        new_landmarks.append(old_landmarks[index])
        new_filelist.append(ipath)
        process_dedup(ipath)
        continue
      # process new image
      count=count+1
      scores=facemodel_server_predict(p,ipath)
      if scores is None:
        failed.add(ipath)
      else:
        try:
          landmarks,_=alignface.detect_landmarks(ipath,face_d,face_p)
          new_scores.append(scores[0])
          new_landmarks.append(landmarks)
          new_filelist.append(ipath)
        except alignface.FitError:
          failed.add(ipath)
      if ipath not in failed:
        process_dedup(ipath)
        print('{} of {}, {}'.format(count,len(T),ipath))
      else:
        print('{} of {}, {} failed'.format(count,len(T),ipath))
    facemodel_server_stop(p)
  finally:
    facemodel_server_finally(p)
  new_scores=numpy.asarray(new_scores)
  new_landmarks=numpy.asarray(new_landmarks)
  print('{} images, {}, {}'.format(len(new_filelist),new_scores.shape,new_landmarks.shape))
  assert len(new_filelist)==len(new_scores)
  assert len(new_filelist)==len(new_landmarks)
  with open('datasets/facemodel/attributes.npz','wb') as f: numpy.savez(f,scores=new_scores,landmarks=new_landmarks,filelist=new_filelist)
  dedup_descriptors.write('datasets/facemodel/history.npz')
  if interactive:
    dellist=list(dellist)
    if len(dellist)>0:
      print(' '.join(dellist))
      if raw_input('Type "yes" to automatically remove {} images: '.format(len(dellist)))=='yes':
        for x in dellist:
          if os.path.exists(x): os.unlink(x)
        print('Images deleted, need to rebuild dataset again.')
    failed=list(failed)
    if len(failed)>0:
      print(' '.join(failed))
      if raw_input('Type "yes" to automatically remove {} images: '.format(len(failed)))=='yes':
        for x in failed:
          if os.path.exists(x): os.unlink(x)
        print('Images deleted, need to rebuild dataset again.')


def build_url_list():
  # list celeba, helen, megaface images
  # lookup original URLs for LFW-searched google images
  filelist=[x.strip() for x in open('datasets/facemodel/filelist.txt').readlines()]
  #for x in [z for z in filelist if z.startswith('celeba')]:
  #  print(x)
  #for x in [z for z in filelist if z.startswith('helen')]:
  #  print(x)
  #for x in [z for z in filelist if z.startswith('megaface')]:
  #  print(x)
  with open('/data2/lfwgoogleface1/config.pickle','rb') as f:
    config1=pickle.load(f)
  with open('/data2/lfwgoogleface2/config.pickle','rb') as f:
    config2=pickle.load(f)
  missing=0
  for x in [z for z in filelist if z.startswith('lfwgoogle')]:
    y=x.replace('lfwgoogle','images')
    if y in config1['original_url']:
      o=config1['original_url'][y]
      print(x,o)
    elif y in config2['original_url']:
      o=config2['original_url'][y]
      print(x,o)
    else:
      missing=missing+1
      continue
  #print('No URL for {} images.'.format(missing))

if __name__=='__main__':
  rebuild_dataset(interactive=True)
  #build_url_list()

