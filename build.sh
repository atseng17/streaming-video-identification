#!/bin/bash
dockerfile=$(pwd)"/dockerfiles/Dockerfile" 
docker build -t encompass_image -f $dockerfile .