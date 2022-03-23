from table2txt.graph_strategy.rel_graph import RelationGraph

def get_strategy(name):
    if name == 'RelationGraph':
        return RelationGraph()
    else:
        raise ValueError('Stategy (%s) Not supported.' % name)

