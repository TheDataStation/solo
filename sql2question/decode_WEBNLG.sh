#!/bin/bash

if [ "$#" -lt 6 ]; then
  echo "./decode_WEBNLG.sh <model> <checkpoint> <gpu_id> <dataset> <experiment name> <data_part>"
  exit 2
fi

if [[ ${1} == *"bart"* ]]; then
  bash webnlg/test_graph2text_bart.sh ${1} ${2} ${3}
fi
if [[ ${1} == *"t5"* ]]; then
  bash webnlg/test_graph2text.sh ${1} ${2} ${3} ${4} ${5} ${6}
fi








