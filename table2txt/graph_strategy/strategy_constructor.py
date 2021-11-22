from table2txt.graph_strategy.complete_graph import CompleteGraph
from table2txt.graph_strategy.graph_no_caption import GraphNoCaption
from table2txt.graph_strategy.template_graph import TemplateGraph

def get_strategy_lst():
    stg_lst = []
    stg_1 = CompleteGraph()
    stg_lst.append(stg_1)

    #stg_2 = GraphNoCaption()
    #stg_lst.append(stg_2)
    
    return stg_lst

def get_strategy(name):
    if name == 'CompleteGraph':
        return CompleteGraph()
    elif name == 'GraphNoCaption':
        return GraphNoCaption()
    elif name == 'TemplateGraph':
        return TemplateGraph()
    else:
        raise ValueError('Stategy (%s) Not supported.' % name)

