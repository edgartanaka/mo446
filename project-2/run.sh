#!/usr/bin/env bash
#docker run -it --rm --env DISPLAY=$DISPLAY --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v=$(pwd)/..:$(pwd)/.. -w=$(pwd) adnrv/opencv make clean
#docker run -it --rm --env DISPLAY=$DISPLAY --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v=$(pwd)/..:$(pwd)/.. -w=$(pwd) adnrv/opencv make
docker run -it --rm --env DISPLAY=$DISPLAY --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v=$(pwd)/..:$(pwd)/.. -w=$(pwd) adnrv/opencv make main