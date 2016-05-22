#!/bin/sh
cur=$(dirname $0)
pushd $cur

program=com.rikima.dnn.LenetMnistExample

sbt "run-main $program $*"

popd