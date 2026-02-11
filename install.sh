#!/bin/bash
sudo apt-get update
sudo apt-get install -y unzip cmake build-essential libboost-python-dev
cd mot3d
pip3 install -r requirements.txt
rm -rf muSSP-master
unzip mussp-master-6cf61b8.zip
pushd mot3d/solvers/wrappers/muSSP
./cmake_and_build.sh
popd
cd ../
export PYTHONPATH="$PWD/mot3d:$PYTHONPATH"
echo $PYTHONPATH
pip3 install -r requirements.txt