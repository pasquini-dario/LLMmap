import os
from transformers import AutoTokenizer, AutoModel
import torch
import math

CACHE_DIR = os.environ.get('HF_MODEL_CACHE', None)
    
class Embedding:
    def __init__(self, model_name, device_map="auto", model_kargs={}):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR, device_map=device_map, **model_kargs)
        self.max_length = 512
        
    def get_embs(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embedding(self, s):
        prompts_tok = self.tokenizer(s, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=self.max_length).to(self.model.device)
        emb = self.get_embs(self.model(**prompts_tok), prompts_tok.attention_mask)
        return emb.detach().cpu().numpy()
    
    def get_embedding_batched(self, s, batch_size):
        n = len(s)
        num_batches = math.ceil(n/batch_size)
        
        outputs = []
        for i in range(num_batches):
            out_i = self.get_embedding(s[i*batch_size:(i+1)*batch_size])
            outputs.append(out_i)
            
        outputs = torch.concat(outputs)
        return outputs
        
    
EMBEDDING_MODELS = [
    ('intfloat/multilingual-e5-large-instruct', Embedding),
]

def load_model(model_id):
    model_name, model_class = EMBEDDING_MODELS[model_id]
    model = model_class(model_name)
    return model