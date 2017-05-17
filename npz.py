#!/usr/bin/env python3

import numpy
import collections
import os.path

'''
NpzLog is an append-only log which implements a persistent key-value
store for numpy ndarrays. Internally it consists of three parts: an
initial ndarray, a pre-allocated ndarray buffer and a python list. The
key-value store is made persistent with an npz file. Suppose d is the
dimensionality of the data, dtype is the data type and b is the number
of preallocated entries. Typical initialization is

if os.path.exists('foo.npz'):
  foo=npz.NpzLog('foo.npz',shape=(0,d),dtype=dtype,buflen=b)
else:
  foo=npz.NpzLog(None,shape=(0,d),dtype=dtype,buflen=b)

Example usage is

bar=foo[key] # recall ndarray
foo[key]=baz # store ndarray
foo.write('foo.npz') # persist

where bar and baz are ndarrays of shape (d,). Unlike ndarray.append,
NpzLog is no-copy when appending new entries. Instead the pre-allocated
buffer is used until it is filled then additional entries are stored in
a python list (which is less space-efficient). When NpzLog is written
to disk it will make a compacted copy of data.

Note: del foo[key] will work, but saving an NpzLog with holes is not
implemented.
'''

class NpzLog(collections.MutableMapping):
  def __init__(self,filename,dtype=numpy.float64,shape=None,buflen=0):
    if filename is None:
      self._k=[]
      assert shape[0]==0
      self._v=numpy.empty(shape,dtype=dtype)
      self._k_map={}
    else:
      data=numpy.load(filename)
      self._k=list(data['k'])
      self._v=data['v']
      self._k_map={k:i for i,k in enumerate(self._k)}
    assert len(self._k)==len(self._v)
    buffer_shape=list(self._v.shape)
    buffer_shape[0]=buflen
    self._bv=numpy.empty(buffer_shape,dtype=self._v.dtype)
    self._tail=len(self._v)
    self._n=len(self._v)
    self._m=len(self._v)+len(self._bv)
    self._ov=[]
    self._holes=False
  def __getitem__(self,key):
    index=self._k_map[key]
    if index<self._n:
      return self._v[index]
    elif index<self._m:
      return self._bv[index-self._n]
    else:
      return self._ov[index-self._m]
  def __setitem__(self,key,value):
    if key in self._k_map:
      index=self._k_map[key]
      if index<self._n:
        self._v[index]=value
      elif index<self._m:
        self._bv[index-self._n]=value
      else:
        self._ov[index-self._m]=value
    else:
      index=self._tail
      if index<self._m:
        self._bv[index-self._n]=value
      else:
        self._ov.append(value)
      self._k.append(key)
      self._k_map[key]=index
      self._tail=self._tail+1
  def __delitem__(self,key):
    self._holes=True
    del self._k_map[key]
  def __iter__(self):
    return iter(self._k_map)
  def __len__(self):
    return len(self._k_map)
  def flatten(self):
    if self._holes:
      raise NotImplementedError
    else:
      if self._tail>self._m:
        return list(self._k),numpy.concatenate([self._v,self._bv,self._ov],axis=0)
      elif self._tail>self._n:
        return list(self._k),numpy.concatenate([self._v,self._bv[0:self._tail-self._n]],axis=0)
      else:
        return list(self._k),self._v
  def write(self,filename):
    k,v=self.flatten()
    with open(filename,'wb') as f:
      numpy.savez(f,k=k,v=v)
  def usage(self):
    return {'n_storage':self._n, 'n_buffer':min(self._tail-self._n,len(self._bv)), 'n_overflow':len(self._ov)}

