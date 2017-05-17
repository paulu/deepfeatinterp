
import numpy
import os.path
import subprocess
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

def read(ipath,dtype=numpy.float32):
  '''
  Returns a H x W x 3 RGB image in the range of [0,1].
  '''
  img=PIL.Image.open(ipath)
  if img.mode!='RGB':
    img=img.convert('RGB')
  return numpy.asarray(img,dtype=dtype)/255.0

def write(opath,I,**kwargs):
  '''
  Given a H x W x 3 RGB image it is clipped to the range [0,1] and
  written to an 8-bit image file.
  '''
  img=PIL.Image.fromarray((I*255).clip(0,255).astype(numpy.uint8))
  ext=os.path.splitext(opath)[1]
  if ext=='.jpg':
    quality=kwargs['quality'] if 'quality' in kwargs else 95
    img.save(opath,quality=quality,optimize=True)
  elif ext=='.png':
    img.save(opath)
  else:
    # I do not want to save unknown extensions because there is no
    # expectation that the default save options are reasonable.
    raise ValueError('Unknown image extension ({})'.format(ext))

def resize(I,shape):
  image=PIL.Image.fromarray((I*255).clip(0,255).astype(numpy.uint8))
  if shape[0]<I.shape[0] and shape[1]<I.shape[1]:
    image=image.resize((shape[1],shape[0]),PIL.Image.BICUBIC)
  else:
    image=image.resize((shape[1],shape[0]),PIL.Image.LANCZOS)
  return numpy.asarray(image)/255.0

def scale(I,scale_factor):
  return resize(I,(int(round(I.shape[0]*scale_factor)),int(round(I.shape[1]*scale_factor))))

def resize_to_fit(I,shape):
  scale_factor=min(float(shape[0])/I.shape[0],float(shape[1])/I.shape[1])
  return scale(I,scale_factor)

def resize_to_cover(I,shape):
  scale_factor=max(float(shape[0])/I.shape[0],float(shape[1])/I.shape[1])
  return scale(I,scale_factor)

def montage(M,sep=0,canvas_value=0):
  # row X col X H X W X C
  assert M.ndim==5
  canvas=numpy.ones((M.shape[0]*M.shape[2]+(M.shape[0]-1)*sep,M.shape[1]*M.shape[3]+(M.shape[1]-1)*sep,M.shape[4]),dtype=M.dtype)*canvas_value
  for i in range(M.shape[0]):
    for j in range(M.shape[1]):
      canvas[i*(M.shape[2]+sep):i*(M.shape[2]+sep)+M.shape[2],j*(M.shape[3]+sep):j*(M.shape[3]+sep)+M.shape[3]]=M[i,j]
  return canvas

def montage_fixed_width(S):
  '''
  S is a list of N rows. Each row is a list of H x W x 3 images.

  Returns a rectangular image where each row is the same width.
  '''
  Hmean=[sum(y.shape[0] for y in x)//len(x) for x in S] # mean height of unscaled rows
  scale=[[Hmean[i]/float(y.shape[0]) for y in x] for i,x in enumerate(S)] # scale equalizes image height in each row
  Wsum=[sum(y.shape[1]*scale[i][j] for j,y in enumerate(x)) for i,x in enumerate(S)] # width of height-scaled rows
  Wmean=int(sum(Wsum)//len(S)) # mean width
  scale2=[[Wmean/float(Wsum[i])*scale[i][j] for j,y in enumerate(x)] for i,x in enumerate(S)] # scale equalizes image height in each row and row width
  Hmean2=[int(sum(y.shape[0]*scale2[i][j] for j,y in enumerate(x))//len(x)) for i,x in enumerate(S)] # mean height of unscaled rows
  shape=[[(Hmean2[i],int(y.shape[1]*scale2[i][j]),3) for j,y in enumerate(x)] for i,x in enumerate(S)] # new height and idth for each image
  Wsum2=[sum(shape[i][j][1] for j,y in enumerate(x)) for i,x in enumerate(S)] # width of height-scaled rows
  H=sum(Hmean2)
  result=numpy.empty((H,Wmean,3),dtype=S[0][0].dtype)
  def resize(a,shape):
    if a.dtype==numpy.uint8:
      img=PIL.Image.fromarray(a)
    else:
      img=PIL.Image.fromarray((a*255).astype(numpy.uint8))
    if shape[1]<img.size[0] or shape[0]<img.size[1]:
      img=img.resize((shape[1],shape[0]),PIL.Image.LANCZOS)
    else:
      img=img.resize((shape[1],shape[0]),PIL.Image.BILINEAR)
    if a.dtype==numpy.uint8:
      result=numpy.array(img)
    else:
      result=(numpy.array(img)/255.0).astype(a.dtype)
    return result
  idx0=0
  for i,x in enumerate(S):
    idx1=0
    for j,y in enumerate(x):
      if j==len(x)-1:
        # absorb rounding errors into last image of each row
        result[idx0:idx0+shape[i][j][0],idx1:]=resize(S[i][j],(shape[i][j][0],result.shape[1]-idx1))
      else:
        result[idx0:idx0+shape[i][j][0],idx1:idx1+shape[i][j][1]]=resize(S[i][j],shape[i][j])
      idx1+=shape[i][j][1]
    idx0+=shape[i][0][0]
  return result

try:
  def render_text(s,size,font=PIL.ImageFont.truetype(subprocess.check_output(['fc-match','-f','%{file}','Droid Serif']),80)):
    '''
    Returns rendered text as a H x W x 3 numpy array in the range [0,1]. The
    image will be the exact size given. If one dimension is None then the
    image will be sized to fit the given dimension.
    '''
    image=PIL.Image.new('RGB',(5,5),(255,255,255))
    draw=PIL.ImageDraw.Draw(image)
    image=PIL.Image.new('RGB',draw.textsize(s,font=font),(255,255,255))
    draw=PIL.ImageDraw.Draw(image)
    draw.text((0,0),s,(0,0,0),font=font)
    if size[0]==None:
      size=list(size)
      size[0]=int(round(image.size[1]*size[1]/float(image.size[0])))
    elif size[1]==None:
      size=list(size)
      size[1]=int(round(image.size[0]*size[0]/float(image.size[1])))
    image=image.resize((size[1],size[0]),PIL.Image.LANCZOS)
    I=numpy.array(image)/255.0
    return I
except:
  def render_text(*args,**kwargs):
    '''
    Returns rendered text as a H x W x 3 numpy array in the range [0,1]. The
    image will be the exact size given. If one dimension is None then the
    image will be sized to fit the given dimension.
    '''
    raise RuntimeError('Could not define render_text (probably because fc-match failed).')

def concatenate(X,axis,canvas_value=0,gravity=(-1)):
  '''
  Given a sequence of images, concatenate them along the given axis,
  expanding the other axes as needed. If gravity is zero then the original
  data will be centered in the output domain. Negative or positive gravity
  will cause it to be flush with the lower or upper bound, respectively.
  '''
  outshape=[sum(x.shape[i] for x in X) if i==axis else max(x.shape[i] for x in X) for i in range(X[0].ndim)]
  Y=[]
  for x in X:
    newshape=list(outshape)
    newshape[axis]=x.shape[axis]
    if gravity>0:
      Y.append(numpy.pad(x,[(newshape[i]-x.shape[i],0) for i in range(x.ndim)],'constant',constant_values=canvas_value))
    elif gravity==0:
      Y.append(numpy.pad(x,[((newshape[i]-x.shape[i])//2,(newshape[i]-x.shape[i])-(newshape[i]-x.shape[i])//2) for i in range(x.ndim)],'constant',constant_values=canvas_value))
    else:
      Y.append(numpy.pad(x,[(0,newshape[i]-x.shape[i]) for i in range(x.ndim)],'constant',constant_values=canvas_value))
  return numpy.concatenate(Y,axis=axis)

