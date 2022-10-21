#!/bin/bash

if [ "$#" -lt 6 ]; then
  echo "./decode_sql2nlg.sh <model> <checkpoint> <gpu_id> <dataset> <experiment name> <data_part>"
  exit 2
fi

if [[ ${1} == *"bart"* ]]; then
  bash sql2nlg/test_sql2nlg_bart.sh ${1} ${2} ${3}
fi
if [[ ${1} == *"t5"* ]]; then
  bash sql2nlg/test_sql2nlg.sh ${1} ${2} ${3} ${4} ${5} ${6}
fi








