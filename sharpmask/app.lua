require 'torch'
require 'cutorch'
require 'image'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('evaluate deepmask/sharpmask')
cmd:text()
cmd:argument('-model', 'path to model to load')
cmd:text('Options:')
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

function generateMasked ( img )
  local h,w = img:size(2),img:size(3)
  infer:forward(img)
  local masks,_ = infer:getTopProps(.2,h,w)
  local res = img:clone()
  drawMasks(res, masks, 10)
  return res
end

--------------------------------------------------------------------------------
-- start server
local app = require('waffle')

app.get('/', function(req, res)
   res.send(html {
     head {
       link {
         rel = 'stylesheet',
         href = 'https://cdnjs.cloudflare.com/ajax/libs/bulma/0.6.0/css/bulma.min.css'
       }
     },
     body {
       div {
         class = 'section',
         div {
           class = 'container',
           h1 {
             class = 'title',
             'Choose image to mask'
           },
           form {
             action = '/',
             method = 'POST',
             enctype = 'multipart/form-data',
             div {
               class = 'field',
               div {
                 class = 'file',
                 label {
                   class = 'file-label',
                   input {
                     class = 'file-input',
                     type = 'file',
                     name = 'file'
                   },
                   span {
                     class = 'file-cta',
                     span {
                       class = 'file-label',
                       'Choose a file...'
                     }
                   },
                   span {
                     class = 'file-name',
                     'No file chosen'
                   }
                 }
               }
             },
             div {
               class = 'field',
               div {
                 class = 'control',
                 button {
                   class = 'button is-link is-primary',
                   'Mask'
                 }
               }
             }
           }
         }
       },
       script {
         [[
         var file = document.querySelector('.file-input');
         file.addEventListener('change', function () {
           if (file.files.length > 0) {
             document.querySelector('.file-name').innerHTML = file.files[0].name;
           }
         });
         ]]
       }
     }
   })
end)

app.post('/', function(req, res)
  local img = req.form.file:toImage()
  local masked = generateMasked(img)
  image.save('./res.jpg', masked)
  res.header('Content-Type', 'image/jpeg')
    .header('Content-Disposition', 'attachment; filename="mask.jpg"')
    .sendFile('./res.jpg')
end)

app.listen({host='0.0.0.0', port=8888})
