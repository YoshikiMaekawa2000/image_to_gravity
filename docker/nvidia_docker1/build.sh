#!/bin/bash

image_name="image_to_gravity"
tag_name="nvidia_docker1"

docker build . \
	-t $image_name:$tag_name \
	--build-arg CACHEBUST=$(date +%s)
