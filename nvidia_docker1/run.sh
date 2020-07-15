#!/bin/bash

pkg_name="image_to_gravity"
root_path=$(pwd)
echo $root_path

xhost +
nvidia-docker run -it --rm \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--net=host \
	-v $root_path/../dataset:/home/$pkg_name/dataset \
	$pkg_name:nvidia_docker1
