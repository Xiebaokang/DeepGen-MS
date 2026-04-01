#!/bin/bash

if [ ! -d "./build" ]; then
  ./build.sh
else
  cd ./build
  ninja -j32

  so_files=(*.so)
  if [ ! -d "../DeepGen" ]; then
    mkdir ../DeepGen
  fi
  cp ./${so_files[0]} ../DeepGen
  cd ..
fi