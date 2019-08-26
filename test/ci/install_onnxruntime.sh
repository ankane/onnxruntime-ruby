#!/usr/bin/env bash

set -e

CACHE_DIR=$HOME/onnxruntime/$ONNXRUNTIME_VERSION

if [ ! -d "$CACHE_DIR" ]; then
  wget https://github.com/microsoft/onnxruntime/releases/download/v$ONNXRUNTIME_VERSION/onnxruntime-linux-x64-$ONNXRUNTIME_VERSION.tgz
  tar xvfz onnxruntime-linux-x64-$ONNXRUNTIME_VERSION.tgz
  mv onnxruntime-linux-x64-$ONNXRUNTIME_VERSION $CACHE_DIR
else
  echo "ONNX Runtime cached"
fi
