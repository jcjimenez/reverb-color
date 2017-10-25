-- Based on computeProposals by Facebook - see https://github.com/facebookresearch/deepmask
require 'torch'
require 'cutorch'
require 'image'

--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('evaluate deepmask/sharpmask')
cmd:text()
cmd:argument('-model', 'path to model to load')
cmd:text('Options:')
cmd:option('-img', 'path/to/test/image')
cmd:option('-output', './res.jpg', 'location and filename for output')
cmd:option('-gpu', 1, 'gpu device')

local config = cmd:parse(arg)

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)

local coco = require 'coco'
maskApi = coco.MaskApi

local meanstd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}

--------------------------------------------------------------------------------
-- load moodel
paths.dofile('DeepMask.lua')
paths.dofile('SharpMask.lua')

local m = torch.load(config.model)
local model = m.model
model:inference(5)
model:cuda()

--------------------------------------------------------------------------------
-- create inference module
local scales = {}
for i = -2.5,.5,.5 do table.insert(scales,2^i) end

if torch.type(model)=='nn.DeepMask' then
  paths.dofile('InferDeepMask.lua')
elseif torch.type(model)=='nn.SharpMask' then
  paths.dofile('InferSharpMask.lua')
end

local infer = Infer{
  np = 5,
  scales = scales,
  meanstd = meanstd,
  model = model,
  dm = false,
}

print('| start')

-- load image
local img = image.load(config.img)
local h,w = img:size(2),img:size(3)

-- forward all scales
infer:forward(img)

function drawMasks ( img, masks )
  --- merge masks
  local rs = maskApi.encode(masks)
  local encoded = maskApi.merge(rs)
  local merged = maskApi.decode(encoded)

  assert(img:isContiguous() and img:dim()==3)
  local n, h, w = merged:size(1), merged:size(2), merged:size(3)
  for i=1,n do
    local M = merged[i]:contiguous():data()
    -- white out any part of image not in mask
    for j=1,3 do
      local O = img[j]:data()
      for k=0,w*h-1 do if M[k]==0 then O[k]=1 end end
    end
  end
end

-- get top propsals
local masks,_ = infer:getTopProps(.2,h,w)
local res = img:clone()
drawMasks(res, masks, 10)
image.save(config.output,res)
print('| done')
collectgarbage()
