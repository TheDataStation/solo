import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
from .model import RetrieverConfig

class BertEncoder(nn.Module):
    def __init__(self, config, name):
        super().__init__()
        self.name = name
        self.model = transformers.BertModel(config, add_pooling_layer=False) 

    def embed_text(self, text_ids, text_mask, apply_mask=False, extract_cls=False):
        text_output = self.model(
            input_ids=text_ids,
            attention_mask=text_mask if apply_mask else None
        )
        if type(text_output) is not tuple:
            text_output.to_tuple()
        text_output = text_output[0]

        if extract_cls:
            text_output = text_output[:, 0]
        else:
            if apply_mask:
                text_output = text_output.masked_fill(~text_mask[:, :, None], 0.)
                text_output = torch.sum(text_output, dim=1) / torch.sum(text_mask, dim=1)[:, None]
            else:
                text_output = torch.mean(text_output, dim=1)
        return text_output


class StudentRetriever(transformers.PreTrainedModel):

    config_class = RetrieverConfig
    base_model_prefix = "retriever"

    def __init__(self, config, teacher_state_dict=None):
        super().__init__(config)
        assert config.projection or config.indexing_dimension == 768, \
            'If no projection then indexing dimension must be equal to 768'
        self.teacher = None
        self.config = config
        
        self.question_encoder = BertEncoder(config, 'question') 
        self.ctx_encoder = BertEncoder(config, 'passage') 
       
        if teacher_state_dict is not None: 
            self.copy_teacher_weights(teacher_state_dict)
        
        StudentRetriever.prune_layers(self.ctx_encoder) 
        
    def copy_teacher_weights(self, teacher_weights):
        self.question_encoder.load_state_dict(teacher_weights)
        self.ctx_encoder.load_state_dict(teacher_weights)
   
    @staticmethod
    def prune_layers(encoder):
        updated_layer = nn.ModuleList()
        for idx, module in enumerate(encoder.model.encoder.layer):
            if idx == 0:
                updated_layer.append(module)
                break
        encoder.model.encoder.layer = updated_layer
    
    def set_teacher(self, teacher):
        self.teacher = teacher
         
    def forward(self,
                question_ids,
                question_mask,
                passage_ids,
                passage_mask,
                score_only=False,
                pos_idxes_per_question=None
        ):
        question_output = self.question_encoder.embed_text(
            text_ids=question_ids,
            text_mask=question_mask,
            apply_mask=self.config.apply_question_mask,
            extract_cls=self.config.extract_cls,
        )
        passage_output = self.ctx_encoder.embed_text(
            text_ids=passage_ids,
            text_mask=passage_mask,
            apply_mask=self.config.apply_passage_mask,
            extract_cls=self.config.extract_cls,
        )
        score = torch.matmul(question_output, passage_output.t())
        if score_only:
            return score

        #score = score / np.sqrt(question_output.size(-1))
        softmax_scores = F.log_softmax(score, dim=1)
        loss = F.nll_loss(
            softmax_scores,
            pos_idxes_per_question,
            reduction="mean",
        )
        _, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = ((max_idxs == pos_idxes_per_question).sum())
        return score, loss, correct_predictions_count


    def calc_logits(self, score, temperature):
        soft_score = score / temperature
        logits = F.log_softmax(soft_score, dim=1)
        return logits            

