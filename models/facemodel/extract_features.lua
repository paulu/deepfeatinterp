--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

-- Basic setup
local opt, checkpoints, DataLoader = require('setup')()

-- Files for training and testing
local loader = DataLoader.create(opt, 'test')
local Tester = require 'runners.test'

-- Testing
local bestModel = torch.load('model.t7')
local tester = Tester(bestModel, opt, logger, 'test')

local testResults = tester:test(nil, loader)
local resultsFilename = paths.concat(opt.save, 'features.csv')
local resultsFile = io.open(resultsFilename, 'w')

for i = 1, #testResults.filenames do
  resultsFile:write(testResults.filenames[i] .. ',')
  resultsFile:write(table.concat(testResults.features[i], ','))
  resultsFile:write('\n')
end
resultsFile:close()

print('Saved results to ' .. resultsFilename)
