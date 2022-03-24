if [ "$#" -ne 3 ]; then
    echo "Usage: ./merge_text.sh <dataset> <graph experiemnt> <text experiment>"
    exit
fi
dataset=$1
graph_expr=$2
text_expr=$3
python ./merge_graph_text.py \
--dataset ${dataset} \
--graph_expr ${graph_expr} \
--text_expr ${text_expr} \
