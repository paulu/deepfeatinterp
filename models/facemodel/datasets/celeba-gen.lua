local ffi = require 'ffi'

local M = {}

local function toTensors(pathTable)
  -- Convert the generated list to a tensor for faster loading
  local nImages = #pathTable
  local maxLength = torch.Tensor(tablex.map(function(x) return #x end, pathTable)):max() + 1
  local pathTensor = torch.CharTensor(nImages, maxLength):zero()
  for i, path in ipairs(pathTable) do
    ffi.copy(pathTensor[i]:data(), path)
  end
  return pathTensor
end

function M.exec(opt, cacheFile)
  local testImagePathTable = {}

  for i, basename in ipairs(paths.dir(opt.data)) do
    if string.find(basename, '.jpg') then
      table.insert(testImagePathTable, basename)
    end
  end

  print(testImagePathTable)
  local testImagePath = toTensors(testImagePathTable)

  local info = {
    basedir = opt.datasetDir,
    test = {
      imagePath = testImagePath,
      imageClass = torch.LongTensor(testImagePath:size(1)),
      imageAttrs = torch.LongTensor(testImagePath:size(1), opt.nClasses),
    },
  }

  print(" | saving list of images to " .. cacheFile)
  torch.save(cacheFile, info)
end

return M
