local M = {
  Runner = require 'runners.runner',
}
local Tester = torch.class('Tester', 'Runner', M)

function Tester:__init(model, opt, logger, setName)
  self.model = model
  self.opt = opt
  self.logger = logger
  self.setName = setName or 'Test'
end

function Tester:test(epoch, dataloader)
  -- Computes the top-1 and top-5 err on the validation set

  local timer = torch.Timer()
  local dataTimer = torch.Timer()
  local size = dataloader:size()

  local nCrops = self.opt.tenCrop and 10 or 1
  print('nCrops',nCrops)
  local top1Sum, top5Sum, timeSum = 0.0, 0.0, 0.0
  local N = 0

  local featuresTable = {}
  local filenameTable = {}

  self.model:evaluate()
  for n, sample in dataloader:run() do
    print('sample.input',sample.input:size(),torch.typename(sample.input),sample.input:min(),sample.input:max())
    print(sample.input[{1,1}]:min(),sample.input[{1,1}]:mean(),sample.input[{1,1}]:max())
    print(sample.input[{1,2}]:min(),sample.input[{1,2}]:mean(),sample.input[{1,2}]:max())
    print(sample.input[{1,3}]:min(),sample.input[{1,3}]:mean(),sample.input[{1,3}]:max())
    local dataTime = dataTimer:time().real

    -- Copy input and target to the GPU
    self:copyInputs(sample)
    local output = self.model:forward(self.input)
    local batchSize = output:size(1) / nCrops
    for i = 1, batchSize do
      table.insert(filenameTable, sample.filename[i])
      table.insert(featuresTable, self.model.output[i]:totable())
    end

    local time = timer:time().real
    timeSum = timeSum + time
    N = N + batchSize

    local epochString = epoch and ('[' .. epoch .. ']') or ''
    print((' | %s: %s[%d/%d]   Time %.3f  Data %.3f'):format(
      self.setName, epochString, n, size, time, dataTime))

    timer:reset()
    dataTimer:reset()
  end
  self.model:training()

  return {
    filenames = filenameTable,
    features = featuresTable,
  }
end

return M.Tester
