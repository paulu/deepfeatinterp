--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

-- Basic setup
local matio = require 'matio'
local opt, checkpoints, DataLoader = require('setup')()

-- Files for training and testing
local trainLoader, valLoader, testLoader = DataLoader.create(opt)
local Trainer = require 'runners.train'
local Tester = require 'runners.test'
local OpCounter = require 'utils.OpCounter'

if not opt.testOnly then
  -- Create model
  local models = require 'models/init'
  local checkpoint, optimState, logger = checkpoints.latest(opt)
  local model, criterion = models.setup(opt, checkpoint)

  -- The trainer handles the training loop and evaluation on validation set
  local trainer = Trainer(model, criterion, opt, optimState, logger)
  local validator = Tester(model, opt, trainer.logger, 'valid')

  -- Log parameters and number of floating point operations
  local opCounter = OpCounter(model, opt)
  opCounter:count()
  checkpoints.logResults(opt, trainer.logger, {
    nParams = trainer.params:size(1),
    ops = opCounter:total(),
    opsByType = opCounter:byType(),
  })

  local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
  local bestTop1 = math.huge
  local bestTop5 = math.huge
  for epoch = startEpoch, opt.nEpochs do
    -- Train for a single epoch, run model on validation set
    local trainResults = trainer:train(epoch, trainLoader)
    local valResults = validator:test(epoch, valLoader)

    local bestModel = false
    if valResults.top1 < bestTop1 then
      bestModel = true
      bestTop1 = valResults.top1
      bestTop5 = valResults.top5
      print(' * Best model ', valResults.top1, valResults.top5)
    end

    trainer:log(trainResults, valResults)
    checkpoints.save(epoch, model, trainer.optimState, trainer.logger, bestModel, opt)
  end

  print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))
end


-- Testing
local loader = opt.testOnValid and valLoader or testLoader
local bestModel, logger = checkpoints.best(opt)
local tester = Tester(bestModel, opt, logger, 'test')

local trainResults = tester:test(nil, trainLoader)
matio.save(opt.trainScoresFilename, {
  features = trainResults.features,
  logits = trainResults.logits,
  labels = trainResults.labels
})

local testResults = tester:test(nil, loader)
matio.save(opt.testScoresFilename, {
  features = testResults.features,
  logits = testResults.logits,
  labels = testResults.labels
})

if opt.testOnValid then
  checkpoints.logResults(opt, logger, {
    finalValidTop1 = testResults.top1,
    finalValidTop5 = testResults.top5,
  })
else
  checkpoints.logResults(opt, logger, {
    testTop1 = testResults.top1,
    testTop5 = testResults.top5,
  })
end
print(string.format(' * Results top1: %6.3f  top5: %6.3f', testResults.top1, testResults.top5))


--
