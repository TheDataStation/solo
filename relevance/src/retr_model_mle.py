from src.retr_model import FusionRetrModelBase
import torch.nn as nn

class CustomLinear(nn.Linear):
    def forward(self, input_x, sample=False, calculate_log_probs=False):
        return super().forward(input_x)

class RetrModelMLE(FusionRetrModelBase):
    def create_linear_layer(self, in_features, out_features):
        return CustomLinear(in_features, out_features) 
    
    def sample_forward(self, batch_data, fusion_scores, fusion_states, passage_masks,
                sample=False, calculate_log_probs=False, opts=None, num_samples=None):
        return self(batch_data, fusion_scores, fusion_states, passage_masks,
                sample=sample, calculate_log_probs=calculate_log_probs, opts=opts)
