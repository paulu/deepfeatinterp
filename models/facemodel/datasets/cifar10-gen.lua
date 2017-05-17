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
--  http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz
--

local URL = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'

local M = {}

local function convertToTensor(files, split)
  local data, labels

  for _, file in ipairs(files) do
    local m = torch.load(file, 'ascii')
    if not data then
      data = m.data:t()
      labels = m.labels:squeeze()
    else
      data = torch.cat(data, m.data:t(), 1)
      labels = torch.cat(labels, m.labels:squeeze())
    end
  end

  -- This is *very* important. The downloaded files have labels 0-9, which do
  -- not work with CrossEntropyCriterion
  labels:add(1)

  -- Make data contiguous and right size
  data = data:contiguous():view(-1, 3, 32, 32)

  if split then
    local numSamples = labels:size(1)
    local data1 = data:narrow(1, 1, numSamples - split)
    local data2 = data:narrow(1, numSamples - split + 1, split)
    local labels1 = labels:narrow(1, 1, numSamples - split)
    local labels2 = labels:narrow(1, numSamples - split + 1, split)
    return {data = data1, labels = labels1}, {data = data2, labels = labels2}
  else
    return {
      data = data,
      labels = labels,
    }
  end
end

function M.exec(opt, cacheFile)
  print("=> Downloading CIFAR-10 dataset from " .. URL)
  local ok = os.execute('curl ' .. URL .. ' | tar xz -C /tmp')
  assert(ok == true or ok == 0, 'error downloading CIFAR-10')

  print(" | combining dataset into a single file")
  local trainData, valData = convertToTensor({
    '/tmp/cifar-10-batches-t7/data_batch_1.t7',
    '/tmp/cifar-10-batches-t7/data_batch_2.t7',
    '/tmp/cifar-10-batches-t7/data_batch_3.t7',
    '/tmp/cifar-10-batches-t7/data_batch_4.t7',
    '/tmp/cifar-10-batches-t7/data_batch_5.t7',
  }, 5000)
  local testData = convertToTensor({
    '/tmp/cifar-10-batches-t7/test_batch.t7',
  })

  print(" | saving CIFAR-10 dataset to " .. cacheFile)
  local data = {
    train = trainData,
    val = valData,
    test = testData,
  }

  torch.save(cacheFile, data)
  paths.rmall('/tmp/cifar-10-batches-t7', 'yes')
  return data
end

return M
