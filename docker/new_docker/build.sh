#!/bin/bash

image_name="image_to_gravity"
tag_name="docker"

docker build . \
	-t $image_name:$tag_name \
	--build-arg CACHEBUST=$(date +%s)
