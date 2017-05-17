return function()
  -- Basic setup
  require 'torch'
  require 'paths'
  require 'optim'
  require 'nn'
  require 'nngraph'
  require 'cunn'
  require 'cudnn'
  local opts = require 'opts'
  local opt = opts.parse(arg)

  opt.dataset = 'celeba'
  opt.datasetDir = paths.concat(opt.data)
  opt.batchSize = 16
  os.remove(paths.concat(opt.data, opt.dataset .. '.t7'))

  -- Make torch settings
  torch.setdefaulttensortype('torch.FloatTensor')
  torch.setnumthreads(1)
  torch.manualSeed(opt.manualSeed)
  cutorch.manualSeedAll(opt.manualSeed)

  -- Load custom layers and criteria

  --  Get latest checkpoing and data
  local DataLoader = require 'dataloader'

  return opt, checkpoints, DataLoader
end
