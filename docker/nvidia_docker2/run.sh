#!/bin/bash

image_name="image_to_gravity"
tag_name="nvidia_docker2"
root_path=$(pwd)

xhost +
docker run -it --rm \
	--runtime=nvidia \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	-v $root_path/../../../dataset_image_to_gravity:/home/$image_name/../dataset_image_to_gravity \
	-v $root_path/../../weights:/home/$image_name/weights \
	-v $root_path/../../graph:/home/$image_name/graph \
	-v $root_path/../../logs:/home/$image_name/logs \
	$image_name:$tag_name
