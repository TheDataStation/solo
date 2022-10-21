import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FusionRetrLoss(nn.Module):
    def __init__(self):
        super(FusionRetrLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, batch_score, batch_answers, opts=None):
        batch_loss = .0
        batch_num = 0
        
        for b_idx, item_scores in enumerate(batch_score):
            answer_scores = item_scores  
            
            answer_lst = batch_answers[b_idx]
            pos_idxes, neg_idxes = self.get_pos_neg_idxes(answer_lst)
            if (len(pos_idxes) == 0) or (len(neg_idxes) == 0):
                continue
            item_loss = self.compute_loss(answer_scores, pos_idxes, neg_idxes)
           
            batch_loss += item_loss
            batch_num += 1

        if batch_num > 0: 
            loss = batch_loss / batch_num
            if opts is not None:
                reg_score_lst = opts['reg_score']
                teg_loss = torch.stack(reg_score_lst).mean()
                loss += teg_loss
        else:
            loss = None
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

    def compute_loss(self, answer_scores, pos_idxes, neg_idxes):
        assert(len(pos_idxes) > 0)
        assert(len(neg_idxes) > 0)
        score_lst = []
        for pos_idx in pos_idxes:
            idx_lst = [pos_idx] + neg_idxes
            item_score = answer_scores[idx_lst].view(1, -1)
            score_lst.append(item_score)

        batch_item_score = torch.cat(score_lst, dim=0)
        batch_item_labels = torch.zeros(len(pos_idxes)).long().to(answer_scores.device)
        loss = self.ce_loss(batch_item_score, batch_item_labels)
        return loss

