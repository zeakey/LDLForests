#!/bin/bash

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"
echo "Downloading..."
if [ ! -e LDL-datasets.zip ]; then
    wget --no-check-certificate http://7xocv2.dl1.z0.glb.clouddn.com/dataset/LDL-datasets.zip
fi 
unzip LDL-datasets.zip
