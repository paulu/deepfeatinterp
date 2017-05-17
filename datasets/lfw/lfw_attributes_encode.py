
import numpy

'''
Example loading code:

data=numpy.load('lfw_attributes.npz')
fields=list(data['fields'])
filelist=list(data['filelist'])
score=data['score']
del data
'''

with open('lfw_attributes.txt') as f:
  f.readline()
  fields=[x.strip() for x in f.readline().split('\t')][3:]
  print fields
  filelist=[]
  score=[]
  for rawline in f.readlines():
    rawdata=[x.strip() for x in rawline.split('\t')]
    a=rawdata[0].replace(' ','_')
    b=int(rawdata[1])
    filelist.append('{}/{}_{:04}.jpg'.format(a,a,b))
    score.append([float(x) for x in rawdata[2:]])
    assert(len(score[-1])==len(fields))
  score=numpy.asarray(score)
  print filelist[0]
  print score[0]
  with open('lfw_attributes.npz','wb') as f: numpy.savez(f,fields=fields,score=score,filelist=filelist)
