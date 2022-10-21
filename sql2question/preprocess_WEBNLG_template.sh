#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "./preprocess_WEBNLG_template.sh <debug> <dataset_folder>"
  exit 2
fi

debug=$1
if [ $debug = "1" ]; then
    opt="-m pdb"
else
    opt=""
fi

dataset_folder=$2
python ${opt} webnlg/data/generate_template.py ${dataset_folder}



