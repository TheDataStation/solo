import numpy as np

def get_top_metrics(sorted_idxes, sorted_corrects, item, max_top):
    tag_lst = item['tags']
    sorted_tables = [tag_lst[a]['table_id'] for a in sorted_idxes]
    top_table_dict = {}
    top_metrics = []
    for idx, table in enumerate(sorted_tables):
        if table not in top_table_dict:
            top_table_dict[table] = sorted_corrects[idx]
            top_metrics.append(sorted_corrects[idx])
    return top_metrics[:max_top] 

class MetricRecorder:
    def __init__(self, max_top_lst):
        self.max_top_lst = max_top_lst
        self.reset()

    def reset(self):
        self.N = 0
        self.metric_dict = {}
        for max_top in self.max_top_lst:
            self.metric_dict[max_top] = {'metric_sum':0}

    def update(self, metric_lst):
        self.N += 1
        for max_top in self.max_top_lst:
            metric = max(metric_lst[:max_top])
            metric_sum = self.metric_dict[max_top]['metric_sum']
            self.metric_dict[max_top]['metric_sum'] = metric_sum + metric 

    def get_mean(self):
        for max_top in self.max_top_lst:
            self.metric_dict[max_top]['metric_mean'] = 0
        if self.N > 0:
            for max_top in self.max_top_lst:
                metric_sum = self.metric_dict[max_top]['metric_sum']
                self.metric_dict[max_top]['metric_mean'] =100 * metric_sum / self.N
        return self.metric_dict


