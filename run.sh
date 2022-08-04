#!/bin/bash
docker run --rm -it --shm-size=2000m --gpus all --name encompass_container -v $(pwd):/app encompass_image bash 