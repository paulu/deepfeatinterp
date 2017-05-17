--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This automatically downloads the CIFAR-10 dataset from
--  http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-100-torch.tar.gz
--

local URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'
local valSize = 5000

local M = {}

local function convertCifar100BinToTorchTensor(inputFname)
   local m=torch.DiskFile(inputFname, 'r'):binary()
   m:seekEnd()
   local length = m:position() - 1
   local nSamples = length / 3074 -- 1 coarse-label byte, 1 fine-label byte, 3072 pixel bytes

   assert(nSamples == math.floor(nSamples), 'expecting numSamples to be an exact integer')
   m:seek(1)

   local coarse = torch.ByteTensor(nSamples)
   local fine = torch.ByteTensor(nSamples)
   local data = torch.ByteTensor(nSamples, 3, 32, 32)
   for i=1,nSamples do
      coarse[i] = m:readByte()
      fine[i]   = m:readByte()
      local store = m:readByte(3072)
      data[i]:copy(torch.ByteTensor(store))
   end

   local out = {}
   out.data = data
   -- This is *very* important. The downloaded files have labels 0-9, which do
   -- not work with CrossEntropyCriterion
   out.labels = fine + 1

   return out
end

function M.exec(opt, cacheFile)
  print("=> Downloading CIFAR-100 dataset from " .. URL)
  local ok = os.execute('curl ' .. URL .. ' | tar xz -C /tmp')
  assert(ok == true or ok == 0, 'error downloading CIFAR-100')

  print(" | combining dataset into a single file")

  local trainAndValData = convertCifar100BinToTorchTensor('/tmp/cifar-100-binary/train.bin')
  local testData = convertCifar100BinToTorchTensor('/tmp/cifar-100-binary/test.bin')

  local numSamples = trainAndValData.labels:size(1)
  local trainData = {
    data = trainAndValData.data:narrow(1, 1, numSamples - valSize),
    labels = trainAndValData.labels:narrow(1, 1, numSamples - valSize),
  }
  local valData = {
    data = trainAndValData.data:narrow(1, numSamples - valSize + 1, valSize),
    labels = trainAndValData.labels:narrow(1, numSamples - valSize + 1, valSize),
  }

  print(" | saving CIFAR-100 dataset to " .. cacheFile)
  local data = {
    train = trainData,
    val = valData,
    test = testData,
  }

  torch.save(cacheFile, data)
  paths.rmall('/tmp/cifar-100-binary', 'yes')
  return data
end

return M
