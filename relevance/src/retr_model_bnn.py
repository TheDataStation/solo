from src.retr_model import FusionRetrModelBase
from src.bnn.bayesian_linear import BayesianLinear
import torch.nn

class RetrModelBNN(FusionRetrModelBase):
    def set_prior(self, prior):
        self.passage_fnt.set_prior(prior['passage_fnt'] if prior is not None else None)
        self.table_fnt.set_prior(prior['table_fnt'] if prior is not None else None)
        self.feat_l1.set_prior(prior['feat_l1'] if prior is not None else None)
        self.feat_l2.set_prior(prior['feat_l2'] if prior is not None else None)

    def create_linear_layer(self, in_features, out_features):
        return BayesianLinear(in_features, out_features)
    
    def log_prior(self):
        log_prior_sum = self.passage_fnt.log_prior \
                      + self.table_fnt.log_prior \
                      + self.feat_l1.log_prior \
                      + self.feat_l2.log_prior
        return log_prior_sum

    def log_variational_posterior(self):
        log_posterior = self.passage_fnt.log_variational_posterior \
                      + self.table_fnt.log_variational_posterior \
                      + self.feat_l1.log_variational_posterior \
                      + self.feat_l2.log_variational_posterior
        return log_posterior

    def sample_forward(self, batch_data, fusion_scores, fusion_states, passage_masks, 
                      sample=False, calculate_log_probs=False, opts=None, num_samples=1):
        outputs = []
        log_priors = torch.zeros(num_samples)
        log_variational_posteriors = torch.zeros(num_samples)
        for i in range(num_samples):
            output_item = self(batch_data, fusion_scores, fusion_states, passage_masks, 
                                 sample=True, opts=opts)
            output_item = torch.stack(output_item).unsqueeze(0)
            outputs.append(output_item)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        
        outputs = torch.cat(outputs, dim=0).mean(dim=0)
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        
        opts['log_prior'] = log_prior
        opts['log_variational_posterior'] = log_variational_posterior
        return outputs 
        
