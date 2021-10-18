from table2txt.graph_strategy.complete_graph import CompleteGraph
from table2txt.graph_strategy.graph_no_caption import GraphNoCaption
from table2txt.graph_strategy.simple_graph import SimpleGraph

def get_strategy_lst():
    stg_lst = []
    stg_1 = CompleteGraph()
    stg_lst.append(stg_1)

    stg_2 = GraphNoCaption()
    stg_lst.append(stg_2)
    
    stg_3 = SimpleGraph()
    stg_lst.append(stg_3)

    return stg_lst
