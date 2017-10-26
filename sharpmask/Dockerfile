FROM kaixhin/cuda-torch:8.0
RUN sudo apt-get update && sudo apt-get install -y wget
RUN mkdir /root/deepmask
RUN git clone https://github.com/facebookresearch/deepmask.git /root/deepmask
RUN mkdir -p /root/deepmask/pretrained
RUN cd /root/deepmask/pretrained && wget https://s3.amazonaws.com/deepmask/models/resnet-50.t7
RUN luarocks install torch
RUN mkdir /root/coco
RUN git clone https://github.com/cocodataset/cocoapi.git /root/coco
RUN cd /root/coco && luarocks make LuaAPI/rocks/coco-scm-1.rockspec
RUN luarocks install image
RUN luarocks install tds
RUN luarocks install json
RUN luarocks install http://raw.githubusercontent.com/baidu-research/warp-ctc/master/torch_binding/rocks/warp-ctc-scm-1.rockspec
RUN luarocks install optim
RUN luarocks install inn
RUN sudo apt-get install -y libcudnn5 libcudnn5-dev
RUN luarocks install https://raw.githubusercontent.com/benglard/htmlua/master/htmlua-scm-1.rockspec
RUN luarocks install https://raw.githubusercontent.com/benglard/waffle/master/waffle-scm-1.rockspec
ENV CUDNN_PATH=/usr/lib/x86_64-linux-gnu/libcudnn.so.5
RUN mkdir /root/data
RUN sed -i "s/  local cat,ann = torch.random(80)/  local cat,ann = torch.random(1)/" /root/deepmask/DataSampler.lua
RUN mkdir /root/deepmask/api
COPY ./createMask.lua /root/deepmask/
COPY ./app.lua /root/deepmask/
WORKDIR /root/deepmask
