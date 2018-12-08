#!/usr/bin/env bash
docker run -it --rm --env DISPLAY=$DISPLAY --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v=$(pwd)/..:$(pwd)/.. -w=$(pwd) adnrv/opencv python src/klt.py