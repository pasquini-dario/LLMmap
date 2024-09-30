import os
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist

from .utility import read_pickle
from .embedding_model import load_model as load_model_emb
from .inference_model_archs import ClassTokenLayer

MAX_ANSWER_CHARS = 650

def load_model(path, verbose=True):
    logs_path = os.path.join(path, 'logs.pickle')
    logs = read_pickle(logs_path)
    if logs['is_open']:
        if verbose:
            print("## Using open-set model ##\n")
        return InferenceModel_open(path, verbose=True)
    else:
        if verbose:
            print("## Using closed-set model ##\n")
        return InferenceModel_closed(path, verbose=True)
    

class InferenceModel:

    def print(self, *args, **kargs):
        if self.verbose:
            print(*args, **kargs)
            
    def __init__(self, path, verbose=True):
        
        logs_path = os.path.join(path, 'logs.pickle')
        self.logs = read_pickle(logs_path)
        
        self.verbose = verbose
        
        self.print("Building Inference Model...")
        
        self.is_open = self.logs['is_open']
        
        self.llms_supported = list(self.logs['label_map'].keys())
        self.label_map = {v:k for (k,v) in self.logs['label_map'].items()}
        self.queries = self.logs['selected_queries']
        
        self.print("\tLoading Inference Model...")
        model_path = os.path.join(path, 'model.keras')
        self.model = tf.keras.models.load_model(model_path, custom_objects={'ClassTokenLayer': ClassTokenLayer})
        
        self.print("\tLoading Embedding Model...")
        self.emb_model_id = self.logs.get('emb_model_id', 0)
        self.emb_model = load_model_emb(self.emb_model_id)
        
        self.print("\tPre-comupting Queries embeddings...")
        self.emb_queries = self.emb_model.get_embedding(self.queries)     
        self.print("Model ready for inference.")
        
        
    def __call__(self, answers):
        if len(answers) != len(self.queries):
            raise Exeception(f"Model supports {self.queries} queries, {len(answers)} answers provided")
        answers = [self._preprocess_answers(answer) for answer in answers]
        emb_outs = self.emb_model.get_embedding(answers) 
        traces = np.concatenate((self.emb_queries, emb_outs), 1)[np.newaxis,:]
        output = self.model(traces, training=False)[0]
        return output
        
    def _preprocess_answers(self, out):
        return out[:MAX_ANSWER_CHARS]
                
class InferenceModel_closed(InferenceModel):

    def __call__(self, answers):
        logits = super().__call__(answers)
        p = tf.nn.softmax(logits).numpy()
        return p
    
    def print_result(self, probabilities, k=5):
        if k < 1:
            raise ValueError("k must be at least 1")
        if k > len(probabilities):
            raise ValueError("k cannot be greater than the number of classes")

        sorted_indices = np.argsort(probabilities)[::-1]
        top_k_indices = sorted_indices[:k]
        top_k_probs = probabilities[top_k_indices]

        print("Prediction:\n")
        for i, (index, prob) in enumerate(zip(top_k_indices, top_k_probs)):
            if prob < 0.001:
                prob_str = f"{prob:.1e}"
            else:
                prob_str = f"{prob:.4f}"

            if i == 0:  # Top-1 class
                print(f"\t[Pr: {prob_str}] \t--> {self.label_map[index]} <--")
            else:
                print(f"\t[Pr: {prob_str}] \t{self.label_map[index]}")
                
class InferenceModel_open(InferenceModel):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.DB_labels, self.DB = self.logs['DB_templates']
        self.distance_fn = self.logs['distance_fn']
        
    def __call__(self, answers):
        emb = super().__call__(answers)
        distances = cdist([emb], self.DB, metric=self.distance_fn)[0]
        return distances

    def print_result(self, distances, k=5):
        if k < 1:
            raise ValueError("k must be at least 1")
        if k > len(distances):
            raise ValueError("k cannot be greater than the number of classes")

        sorted_indices = np.argsort(distances)
        top_k_indices = sorted_indices[:k]
        top_k_probs = distances[top_k_indices]

        print("Prediction:\n")
        for i, (index, dist) in enumerate(zip(top_k_indices, top_k_probs)):
            if dist < 0.001:
                dist_str = f"{dist:.1e}"
            else:
                dist_str = f"{dist:.4f}"

            if i == 0:  # Top-1 class
                print(f"\t[Distance: {dist_str}] \t--> {self.label_map[index]} <--")
            else:
                print(f"\t[Distance: {dist_str}] \t{self.label_map[index]}")

                

        
