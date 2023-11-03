import torch
from torch.nn.functional import log_softmax, softmax
from transformers import LogitsProcessor
from transformers.modeling_outputs import BaseModelOutput

class MoELogitsProcessor(LogitsProcessor):
    def __init__(self, model, mixture_fn, encoder_inputs):
        self.model = model
        self.encoder_outputs = model.get_encoder()(**encoder_inputs.to(model.device), return_dict=True)
        self.mixture_fn = mixture_fn
    
    def __call__(self, input_ids, scores):
        
        # self.encoder_outputs.last_hidden_state.unsqueeze(1).expand(-1, 50, -1, -1).reshape(32 * 50, 11, 1472).shape
        batch_size = self.encoder_outputs.last_hidden_state.shape[0]
        num_samples = input_ids.shape[0] // batch_size
        enc_seq_len = self.encoder_outputs.last_hidden_state.shape[1]
        enc_hidden_size = self.encoder_outputs.last_hidden_state.shape[2]
        encoder_outputs = BaseModelOutput(**{k: v.unsqueeze(1).expand(batch_size, num_samples, -1, -1).reshape(batch_size * num_samples, enc_seq_len, enc_hidden_size) for k, v in self.encoder_outputs.items()})
        
        # import ipdb; ipdb.set_trace()
        outputs = self.model(
            decoder_input_ids=input_ids, 
            encoder_outputs=encoder_outputs, 
            return_dict=True
            )
        mixture = self.mixture_fn([scores, outputs.logits[:, -1, :]])
        return mixture

class AdditiveMixture:
    def __init__(self, alphas):
        assert sum(alphas) == 1, "Mixture weights must sum to 1"
        self.alphas = alphas
    
    def __call__(self, scores):
        return torch.stack(
            [alpha * log_softmax(score, dim=-1) for alpha, score in zip(self.alphas, scores)],
            dim=-2).sum(dim=-2)

class ContrastiveMixture:
    def __init__(self):
        return
    
    def __call__(self, scores):
        expert_scores, amateur_scores = scores[0], scores[1:]

        am_aggregate = torch.stack([log_softmax(am_score, dim=-1) for am_score in amateur_scores], dim=-2).sum(dim=-2)

        return log_softmax(expert_scores, dim=-1) - am_aggregate

