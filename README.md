
# <img height="100" src="https://pasquini-dario.github.io/logo_llmap.png"> LLMmap (Fingerprinting For Large Language Models)

Basic functionalities implemented. Better and wider range of options will be added as the project evolves (sorry for all this latency).

## Requiriments 
```
pip install -r requirements.txt
```

## Download pre-trained models
At the moment, we provide two open and closed-set pre-trained models for the supported LLMs below.

To use the models:
* Download the archive [here](https://drive.google.com/file/d/1byhaQ4VyI1ChxpmsW7mShcZVSn_krOzF/view?usp=share_link).
* ```unzip data.zip``` inside the repository.

To test out the models, use the ```main_interactive.py``` script:

```
usage: main_interactive.py [-h] [--inference_model_path INFERENCE_MODEL_PATH] [--gpus GPUS]

Interactive session for LLM fingeprinting

options:
  -h, --help            show this help message and exit
  --inference_model_path INFERENCE_MODEL_PATH
                        Path inference model to use
  --gpus GPUS           Comma-separated list of GPUs to use (e.g., "0,1,2,3")
```

### Supported LLMs
* CohereForAI/aya-23-35B
* CohereForAI/aya-23-8B
* Deci/DeciLM-7B-instruct
* HuggingFaceH4/zephyr-7b-beta
* NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO
* Qwen/Qwen2-1.5B-Instruct
* Qwen/Qwen2-72B-Instruct
* Qwen/Qwen2-7B-Instruct
* abacusai/Smaug-Llama-3-70B-Instruct
* claude-3-5-sonnet-20240620
* claude-3-haiku-20240307
* claude-3-opus-20240229
* google/gemma-1.1-2b-it
* google/gemma-1.1-7b-it
* google/gemma-2-27b-it
* google/gemma-2-9b-it
* google/gemma-2b-it
* google/gemma-7b-it
* gpt-3.5-turbo
* gpt-4-turbo-2024-04-09
* gpt-4o-2024-05-13
* gradientai/Llama-3-8B-Instruct-Gradient-1048k
* internlm/internlm2_5-7b-chat
* meta-llama/Llama-2-7b-chat-hf
* meta-llama/Meta-Llama-3-70B-Instruct
* meta-llama/Meta-Llama-3-8B-Instruct
* meta-llama/Meta-Llama-3.1-70B-Instruct
* meta-llama/Meta-Llama-3.1-8B-Instruct
* microsoft/Phi-3-medium-128k-instruct
* microsoft/Phi-3-medium-4k-instruct
* microsoft/Phi-3-mini-128k-instruct
* microsoft/Phi-3-mini-4k-instruct
* microsoft/Phi-3.5-MoE-instruct
* mistralai/Mistral-7B-Instruct-v0.1
* mistralai/Mistral-7B-Instruct-v0.2
* mistralai/Mistral-7B-Instruct-v0.3
* mistralai/Mixtral-8x7B-Instruct-v0.1
* nvidia/Llama3-ChatQA-1.5-8B
* openchat/openchat-3.6-8b-20240522
* openchat/openchat_3.5
* togethercomputer/Llama-2-7B-32K-Instruct
* upstage/SOLAR-10.7B-Instruct-v1.0

## Paper
Paper available [here](https://arxiv.org/pdf/2407.15847). To cite it:
```
@misc{pasquini2024llmmapfingerprintinglargelanguage,
      title={LLMmap: Fingerprinting For Large Language Models}, 
      author={Dario Pasquini and Evgenios M. Kornaropoulos and Giuseppe Ateniese},
      year={2024},
      eprint={2407.15847},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2407.15847}, 
}
```
