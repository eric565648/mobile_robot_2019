#!/bin/bash

docker run -it --rm \
       --network=host \
       --privileged \
       --name duckiefloat-nano \
       -e DISPLAY=$DISPLAY \
       -v $HOME/.Xauthority:/root/.Xauthority \
       -v /dev:/dev \
       -v $(pwd)/:/root/mobile_robot_2019 \
       -w /root/mobile_robot_2019 \
       argnctu/subt:arm64v8-duckiefloat
