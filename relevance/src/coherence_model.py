import torch
import torch.nn as nn
import os
from src.data import Collator
import src.evaluation
import copy
from torch.nn import CrossEntropyLoss
from src.data import get_backward_question
import src.data

class ForwardReader:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

    def generate(self, input_ids, attention_mask, batch_examples, num_answers=5):
        opt_info = {}
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=num_answers,
                num_return_sequences=num_answers,
                opt_info=opt_info
            )
        batch_size = len(input_ids)
        assert(len(outputs) == (batch_size * num_answers))
        pred_scores = torch.tensor(opt_info['pred_scores']).view(batch_size, num_answers)
        outputs = outputs.reshape(batch_size, num_answers, -1)
        for b_idx, answer_code_lst in enumerate(outputs):
            example = batch_examples[b_idx]
            src_data_item = example['src_data_item']
            top_pred_info_lst = []
            for answer_idx, answer_code in enumerate(answer_code_lst):
                pred_answer = self.tokenizer.decode(answer_code, skip_special_tokens=True)
                em_score = None
                if 'answers' in src_data_item:
                    em_score = src.evaluation.ems(pred_answer, src_data_item['answers'])
                
                f_pred_info = {
                    'answer':pred_answer,
                    'forward_score':pred_scores[b_idx][answer_idx].item(),
                    'em':int(em_score)
                }
                top_pred_info_lst.append(f_pred_info)
                 
            src_data_item['top_preds'] = top_pred_info_lst

 
class BackwardReader(nn.Module):
    def __init__(self, tokenizer, model):
        super(BackwardReader, self).__init__()
        self.tokenizer = tokenizer
        self.model = model
        #self.model.eval()

    def generate(self, input_ids, attention_mask, batch_examples, opt_info=None):
        #with torch.no_grad():
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=50,
            opt_info=opt_info
        )


class CoherenceCollator(Collator):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        super(CoherenceCollator, self).__init__(text_maxlength, tokenizer, answer_maxlength=answer_maxlength)
    
    def __call__(self, batch):
        outputs = super().__call__(batch)
        coherence_outputs = outputs + (batch,)
        return coherence_outputs


class CoherenceModel(nn.Module):
    def __init__(self, f_reader, b_reader, collator):
        super(CoherenceModel, self).__init__()
        self.f_reader = f_reader
        self.b_reader = b_reader
        self.collator = collator
    
        D = 768
        input_d = D * 3
        self.sub_match_f = nn.Sequential(
                        nn.Linear(input_d, D),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(D, 1)
                        )


    def forward(self, input_ids, attention_mask, labels, batch_examples):
        if not 'top_preds' in batch_examples[0]['src_data_item']: 
            self.f_reader.generate(input_ids, attention_mask, batch_examples)
            #torch.cuda.empty_cache()
             
        loss_item_lst = [] 
        for example in batch_examples:
            src_data_item = example['src_data_item']
            forward_preds = src_data_item['top_preds']
            b_example_lst = []
           
            correct_answers = [a for a in forward_preds if a['em'] > 0]
            if labels is not None:
                if (len(correct_answers) == 0) or (len(correct_answers) == len(forward_preds)):
                    continue
            
            if True: #'coherence_feat' not in src_data_item:
                for f_pred in forward_preds:
                    if 'back_example' not in f_pred:
                        b_example = copy.deepcopy(example)
                        del b_example['src_data_item'] 
                        self.construct_back_info(b_example, src_data_item, f_pred['answer'])
                        f_pred['back_example'] = b_example
                    b_example_lst.append(f_pred['back_example'])
               
                b_batch = self.collator(b_example_lst)
                (_, _, _, context_ids, context_mask, _) = b_batch
                b_input_ids = context_ids.cuda()
                b_attention_mask = context_mask.cuda()
                
                opt_info = {}
                self.b_reader.generate(b_input_ids, b_attention_mask, b_example_lst, opt_info=opt_info)
                
                back_sub_state = opt_info['answer_state']
                back_sub_state = back_sub_state.view(back_sub_state.shape[0], -1) 
                subject_state = self.get_subject_state(opt_info['encoder_outputs'])
                 
                input_feat = torch.cat([subject_state, back_sub_state, subject_state * back_sub_state], dim=1)
                src_data_item['coherence_feat'] = input_feat 
            #else:
            #    input_feat = src_data_item['coherence_feat']
                 
            match_scores = self.sub_match_f(input_feat)
           
            if labels is not None: 
                loss_item = self.compute_loss(match_scores, forward_preds)    
                loss_item_lst.append(loss_item.view(-1))
            else:
                for idx, f_pred in enumerate(forward_preds): 
                    f_pred['coherence_score'] = match_scores[idx].item()


        if labels is not None:
            if len(loss_item_lst) == 0:
                return None

            total_loss = torch.cat(loss_item_lst)
            loss = total_loss.mean()
            return loss
        
             
    def compute_loss(self, match_scores, forward_preds):
        loss_fct = CrossEntropyLoss()
        pos_idxes = []
        neg_idxes = []

        for idx, pred in enumerate(forward_preds):
            if pred['em'] > 0:
                pos_idxes.append(idx)
            else:
                neg_idxes.append(idx)
        
        score_lst = []
        for pos_idx in pos_idxes:
            idx_lst = [pos_idx] + neg_idxes
            item_score = match_scores[idx_lst].view(1, -1)
            score_lst.append(item_score)
        batch_item_score = torch.cat(score_lst, dim=0)
        batch_item_labels = torch.zeros(len(pos_idxes)).long().to(match_scores.device)
        
        loss = loss_fct(batch_item_score, batch_item_labels)
        return loss


    def save_pretrained(self, model_path):
        file_name = 'coherence_model.bin'
        file_path = os.path.join(model_path, file_name)  
        torch.save(self.state_dict(), file_path)

    def get_subject_state(self, encoder_outputs):
        state = encoder_outputs[0].mean(dim=1)
        return state

    def construct_back_info(self, b_example, src_data_item, pred_answer):
        question = src_data_item['question']
        subject = src_data_item['subject']
        target = pred_answer
        back_question = get_backward_question(question, subject, target)
        b_example['target'] = subject + ' </s>'
        b_example['question'] = src.data.Question_Prefix + " " + back_question
        
         
