import os
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types

max_new_tokens = 100
CACHE_DIR = os.environ.get('HF_MODEL_CACHE', None)

class LLM_huggingface:
    
    def __init__(
        self,
        llm_name,
        model_class=AutoModelForCausalLM,
        tokenizer_class=AutoTokenizer,
        model_load_kargs={},
        tokenizer_only=False,
    ):

        api_key = os.environ.get('HUGGINGFACE_API_KEY', None)
        if api_key is None:
            raise Exception(f'Missing HuggingFace APIs key. Export "HUGGINGFACE_API_KEY" in the enverioment and try again')
            
        self.llm_name = llm_name
        self.model_class = model_class
        
        self.tokenizer = tokenizer_class.from_pretrained(llm_name, padding_side='left', token=api_key, legacy=False, **model_load_kargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token            
        self.tokenizer.with_system_prompt = True
        
        self.is_hf = True

        self.model = None
        if not tokenizer_only:
            self.model = model_class.from_pretrained(llm_name, token=api_key, **model_load_kargs)
            self.model.generation_config.pad_token_ids = self.tokenizer.pad_token_id

    @staticmethod
    def _does_template_have_system(tokenizer):
        chat_template = getattr(tokenizer, 'chat_template', None)
        if chat_template is None:
            return False
        return "system" in chat_template

    def make_prompt(self, system, user):
        messages = []
        if system:
            if self._does_template_have_system(self.tokenizer):
                messages.append( {'role':'system', 'content':system} )
            else:
                user = f'{system}\n\n{user}'
                
        messages.append( {'role':'user', 'content':user} )

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return text
            
    def generate(
        self,
        prompt,
        gen_kargs,
        skip_special_tokens=True,
        max_new_tokens=max_new_tokens,
    ):

        with torch.no_grad():
            in_toks = self.tokenizer(
                prompt,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
                return_token_type_ids=False
            ).to(self.model.device)
            out_toks = self.model.generate(
                **in_toks,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **gen_kargs
            )
    
            gen_toks = [out_toks[i,in_toks.input_ids[i].shape[0]:] for i in range(len(out_toks))]
            gen_strs = self.tokenizer.batch_decode(gen_toks, skip_special_tokens=skip_special_tokens)

        return gen_strs


#####################################################################################################################

class LLM_OpenAI:
    def __init__(self, llm_name):
                
        api_key = os.environ.get('OPENAI_API_KEY', None)
        if api_key is None:
            raise Exception(f'Missing OpenAPI APIs key. Export "OPENAI_API_KEY" in the enverioment and try again')
        
        self.client = OpenAI(api_key=api_key)
        self.llm_name = llm_name
        
        self.is_hf = False


    def make_prompt(self, system, user):
        messages = []
        if system:
            messages += [{'role':'system', 'content':system}]
        messages += [{'role':'user', 'content':user}]
        return messages


    def _convert_gen_kargs(self, gen_kargs):
        if 'do_sample' in gen_kargs:
            do_sample = gen_kargs.pop('do_sample')
            if not do_sample:
                gen_kargs['temperature'] = 0

        return gen_kargs

        
    def generate(
        self,
        prompt,
        gen_kargs,
        max_new_tokens=max_new_tokens
    ):

        gen_kargs = self._convert_gen_kargs(gen_kargs)
        gen_kargs['max_tokens'] = max_new_tokens
        
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=prompt,
            **gen_kargs,
        )
        output = response.choices[0].message.content
        return [output]
##################################################################################################################### 


class LLM_Anthropic(LLM_OpenAI):
    def __init__(self, llm_name):

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key is None:
            raise RuntimeError('Missing Anthropic API key. Export "ANTHROPIC_API_KEY" and try again.')
                
        self.client = client = Anthropic(api_key=api_key)
        self.llm_name = llm_name
        self.is_hf = False

    def make_prompt(self, system, user):
        messages = [{'role':'user', 'content':user}]
        return (system, messages)
    
    def generate(
        self,
        prompt,
        gen_kargs,
        max_new_tokens=max_new_tokens
    ):

        system, messages = prompt
        gen_kargs = self._convert_gen_kargs(gen_kargs)
        
        message = self.client.messages.create(
            max_tokens=max_new_tokens,
            system=system, 
            messages=messages,
            model=self.llm_name,
            **gen_kargs,
        )
        
        out = message.content[0].text
        
        return [out]

##################################################################################################################### 

class LLM_Gemini(LLM_OpenAI):
    def __init__(self, llm_name):
        # Gemini often uses GEMINI_API_KEY or GOOGLE_API_KEY
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if api_key is None:
            raise Exception(f'Missing Gemini API key. Export "GEMINI_API_KEY" in the environment.')
        
        self.client = genai.Client(api_key=api_key)
        self.llm_name = llm_name
        self.is_hf = False

    def make_prompt(self, system, user):
        # Gemini SDK uses 'system_instruction' for the system prompt 
        # and a list of contents for the user message.
        messages = [types.Content(role="user", parts=[types.Part.from_text(text=user)])]
        return (system, messages)

    def generate(self, prompt, gen_kargs, max_new_tokens=max_new_tokens):
        system, messages = prompt
        
        # Adjusting gen_kargs to Gemini's expected format (GenerateConfig)
        # Gemini uses 'max_output_tokens' instead of 'max_tokens'
        config = {
            "max_output_tokens": max_new_tokens,
            "temperature": gen_kargs.get("temperature", 0.7 if gen_kargs.get("do_sample") else 0.0)
        }
        
        # If system instruction is provided, it goes into the config or Client call
        response = self.client.models.generate_content(
            model=self.llm_name,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=system,
                **config
            )
        )
        
        return [response.text]

#####################################################################################################################


def load_llm(llm_name, llm_type, cache_dir=CACHE_DIR, **kargs):

    if llm_type == 0:
        kargs['model_load_kargs'] = {'device_map':"auto", 'cache_dir':cache_dir, 'trust_remote_code':True}
        llm = LLM_huggingface(llm_name, **kargs)
    elif llm_type == 1:
        llm = LLM_OpenAI(llm_name)
    elif llm_type == 2:
        llm = LLM_Anthropic(llm_name)
    elif llm_type == 3:
        llm = LLM_Gemini(llm_name)
    else:
        raise Exception()
    return llm