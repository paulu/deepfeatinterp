--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Torch-7 ResNet Training script')
  cmd:text('See https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md for examples')
  cmd:text()
  cmd:text('Options:')
  ------------- File options --------------------
  cmd:option('-data',  os.getenv('DATA_DIR') or '', 'Path to datasets')
  cmd:option('-save',  os.getenv('SAVE_DIR') or '', 'Directory in which to save checkpoints/results')
  ------------- General options --------------------
  cmd:option('-dataset',   'cifar10', 'Options: imagenet | cifar10 | cifar100')
  cmd:option('-manualSeed', 0,       'Manually set RNG seed')
  cmd:option('-nGPU',     1,       'Number of GPUs to use by default')
  cmd:option('-backend',   'cudnn',   'Options: cudnn | cunn')
  cmd:option('-cudnn',    'fastest',  'Options: fastest | default | deterministic')
  ------------- Data options ------------------------
  cmd:option('-nThreads',      2, 'number of data loading threads')
  ------------- Training options --------------------
  cmd:option('-nEpochs',      0,     'Number of total epochs to run')
  cmd:option('-epochNumber',    1,     'Manual epoch number (useful on restarts)')
  cmd:option('-batchSize',     0,    'mini-batch size (1 = pure stochastic)')
  cmd:option('-nTrain',     math.huge,    'Number of training samples')
  cmd:option('-testOnly',      false, 'Run testing only')
  cmd:option('-testOnValid',   false, 'Do final test on validation (rather than test) set')
  cmd:option('-tenCrop',      'false', 'Ten-crop testing')
  ------------- Checkpointing options ---------------
  cmd:option('-resume',       false,  'Resume from the latest checkpoint in this directory')
  ---------- Optimization options ----------------------
  cmd:option('-LR',          0.1,  'initial learning rate')
  cmd:option('-momentum',      0.9,  'momentum')
  cmd:option('-weightDecay',    1e-4,  'weight decay')
  ---------- Model options ----------------------------------
  cmd:option('-netType',    'resnet', 'Options: resnet | preresnet')
  cmd:option('-depth',      0,     'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
  cmd:option('-shortcutType', '',     'Options: A | B | C')
  cmd:option('-transitionType', 'full',     'Options: dense | partial')
  cmd:option('-growthRate', 12,  'How many filters to add each layer')
  cmd:option('-nInitChannels', 16,  'Number of channels to start with initially')
  cmd:option('-dropRate', 0,     'Dropout rate')
  ---------- Model options ----------------------------------
  cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
  cmd:option('-optnet',          'false', 'Use optnet to reduce memory usage')
  cmd:option('-resetClassifier', 'false', 'Reset the fully connected layer for fine-tuning')
  cmd:option('-pretrained', 'none', 'Pretrained')
  cmd:option('-nClasses',      0,    'Number of classes in the dataset')
  cmd:text()

  local opt = cmd:parse(arg or {})

  opt.tenCrop = opt.tenCrop ~= 'false'
  opt.shareGradInput = opt.shareGradInput ~= 'false'
  opt.optnet = opt.optnet ~= 'false'
  opt.resetClassifier = opt.resetClassifier ~= 'false'
  opt.pretrained = (opt.pretrained ~= 'none') and opt.pretrained

  if opt.shareGradInput and opt.optnet then
    cmd:error('error: cannot use both -shareGradInput and -optnet')
  end

  -- Model/dataset specific opts --
  local specificOpts = {}

  if opt.dataset == 'imagenet' then
    specificOpts.shortcutType = 'B'
    specificOpts.nEpochs = 90
    specificOpts.batchSize = 32
    specificOpts.nClasses = 1000

  elseif opt.dataset == 'cifar10' then
    specificOpts.shortcutType = 'A'
    specificOpts.nEpochs = 300
    specificOpts.batchSize = (opt.nettype == 'densenet') and 64 or  256
    specificOpts.nClasses = 10

  elseif opt.dataset == 'cifar100' then
    specificOpts.shortcutType = 'A'
    specificOpts.nEpochs = 300
    specificOpts.batchSize = (opt.nettype == 'densenet') and 64 or  256
    specificOpts.nClasses = 100

  elseif opt.dataset == 'celeba' then
    specificOpts.nEpochs = 150
    specificOpts.batchSize = 256
    specificOpts.nClasses = opt.attr and 40 or 8000

  else
    cmd:error('unknown dataset: ' .. opt.dataset)
  end

  for optKey, specificOptVal in pairs(specificOpts) do
    local origVal = opt[optKey]
    if type(origVal) == 'number' and origVal == 0 then
      opt[optKey] = specificOptVal
    elseif type(origVal) == 'string' and origVal == '' then
      opt[optKey] = specificOptVal
    end
  end

  if opt.resetClassifier then
    if opt.nClasses == 0 then
      cmd:error('-nClasses required when resetClassifier is set')
    end
  end

  -- Filename opts
  assert(paths.dirp(opt.data), 'Invalid data directory')
  if not paths.dirp(opt.save) then
    local res = os.execute('mkdir ' .. opt.save)
    assert(res, 'Could not open or make save directory ' .. opt.save)
  end

  opt.pretrained = opt.pretrained ~= 'none' and opt.pretrained

  opt.modelFilename = paths.concat(opt.save, 'model.t7')
  opt.optimFilename = paths.concat(opt.save, 'optimState.t7')
  opt.loggerFilename = paths.concat(opt.save, 'log.t7')
  opt.latestFilename = paths.concat(opt.save, 'latest.t7')
  opt.trainScoresFilename = paths.concat(opt.save, 'train-scores.mat')
  opt.testScoresFilename = paths.concat(opt.save, 'test-scores.mat')

  opt.datasetDir = paths.concat(opt.data, opt.dataset)

  return opt
end

return M
