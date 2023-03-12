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

class StudentRetriever(transformers.PreTrainedModel):

    config_class = RetrieverConfig
    base_model_prefix = "retriever"

    def __init__(self, config, teacher_model=None):
        super().__init__(config)
        assert config.projection or config.indexing_dimension == 768, \
            'If no projection then indexing dimension must be equal to 768'
        self.teacher = None
        self.config = config
        
        self.question_encoder = BertEncoder(config, 'question') 
        self.ctx_encoder = BertEncoder(config, 'passage') 
       
        if teacher_model is not None: 
            self.copy_teacher_weights(teacher_model)
        
        teacher_model.model.pooler = None
        StudentRetriever.prune_layers(self.ctx_encoder) 
        
  
    def copy_teacher_weights(self, teacher_model):
        teacher_weights = teacher_model.state_dict() 
        for key in ['model.pooler.dense.weight', 'model.pooler.dense.bias']:
            del teacher_weights[key]
             
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
        ):
        question_output = self.embed_text(
            self.question_encoder,
            text_ids=question_ids,
            text_mask=question_mask,
            apply_mask=self.config.apply_question_mask,
            extract_cls=self.config.extract_cls,
        )
        bsz, n_passages, plen = passage_ids.size()
        passage_ids = passage_ids.view(bsz * n_passages, plen)
        passage_mask = passage_mask.view(bsz * n_passages, plen)
        passage_output = self.embed_text(
            self.ctx_encoder,
            text_ids=passage_ids,
            text_mask=passage_mask,
            apply_mask=self.config.apply_passage_mask,
            extract_cls=self.config.extract_cls,
        )

        #batch dot product
        score = torch.einsum(
            'bd,bid->bi',
            question_output,
            passage_output.view(bsz, n_passages, -1)
        )
        
        if score_only:
            return score

        #score = score / np.sqrt(question_output.size(-1))
        softmax_scores = F.log_softmax(score, dim=1)
        pos_idxes_per_question = torch.tensor([0] * score.shape[0]).to(softmax_scores.device)
        loss = F.nll_loss(
            softmax_scores,
            pos_idxes_per_question,
            reduction="mean",
        )
        _, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = ((max_idxs == pos_idxes_per_question).sum())
        return score, loss, correct_predictions_count


    def embed_text(self, encoder, text_ids, text_mask, apply_mask=False, extract_cls=False):
        text_output = encoder.model(
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

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        return self.loss_fct(score, gold_score)
