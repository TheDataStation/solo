import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FusionGeneralRetrLoss(nn.Module):
    def __init__(self):
        super(FusionGeneralRetrLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batch_score, batch_answers, opts=None):
        batch_loss = .0
        batch_num = len(batch_score)
        
        for b_idx, item_scores in enumerate(batch_score):
            answer_lst = batch_answers[b_idx]
            assert(len(item_scores) == len(answer_lst))
            
            pos_idxes, neg_idxes = self.get_pos_neg_idxes(answer_lst)
            
            assert(len(pos_idxes) > 0) and (len(neg_idxes) > 0)

            labels = [(1 if a['em'] >= 1 else 0) for a in answer_lst]
            labels = torch.tensor(labels).float().to(item_scores.device)
            item_loss = self.loss_fn(item_scores, labels)
             
            batch_loss += item_loss

        loss = batch_loss / batch_num
        if opts is not None:
            reg_score_lst = opts['reg_score']
            teg_loss = torch.stack(reg_score_lst).mean()
            loss += teg_loss
       
        assert(loss is not None) 
        return loss

    def get_pos_neg_idxes(self, answer_lst):
        pos_idxes = []
        neg_idxes = []
        for idx, answer in enumerate(answer_lst):
            em_score = answer['em']
            if em_score >= 1:
                pos_idxes.append(idx)
            else:
                neg_idxes.append(idx) 
        return pos_idxes, neg_idxes 


