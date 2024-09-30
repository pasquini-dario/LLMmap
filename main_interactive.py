import os
import argparse
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from LLMmap.utility import set_gpus
from LLMmap.inference import load_model

kb = KeyBindings()

@kb.add('enter')
def accept_input(event):
    event.current_buffer.validate_and_handle()
    
session = PromptSession(
    multiline=True,
    key_bindings=kb,
)


def int_loop(inf):
    # ANSI color codes
    INSTRUCTION_COLOR = '\033[93m'  # Yellow
    QUERY_COLOR = '\033[94m'        # Blue
    PROMPT_COLOR = '\033[92m'       # Green
    RESET_COLOR = '\033[0m'         # Reset color

    # Print the instruction in yellow
    print("\n\n" + INSTRUCTION_COLOR + "[Instruction] Submit the given query to the LLM app and copy/paste the output produced and then ENTER. Let's start:")
    input("[Press any key to continue]: " + RESET_COLOR)
    print("-" * 50)
    
    n = len(inf.queries)
    answers = []
    for i in range(n):
        print('\n\n')
        query = inf.queries[i]
        # Print the query in blue
        print(INSTRUCTION_COLOR + f"[Query to submit ({i+1}/{n})]:\n"+QUERY_COLOR+f"{query}\n" + RESET_COLOR)
        print(INSTRUCTION_COLOR + "[LLM app response]:" + RESET_COLOR, end=' ')
        answer = session.prompt()
        answers.append(answer)
        time.sleep(1)
        
    print(INSTRUCTION_COLOR+"\n\n### RESULTS ###")
    p = inf(answers)
    inf.print_result(p)
    print(RESET_COLOR)
    

if __name__ == "__main__":
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Interactive session for LLM fingeprinting')
    
    parser.add_argument('--inference_model_path', type=str, help='Path inference model to use', default='./data/models/open_set_8q/')
    
    # Define the argument for the list of GPUs
    parser.add_argument('--gpus', type=str, help='Comma-separated list of GPUs to use (e.g., "0,1,2,3")')

    # Parse the arguments
    args = parser.parse_args()
    
    if args.gpus:
        set_gpus(args.gpus)
        
    inf = load_model(args.inference_model_path)
    
    print("\n##### LLMs supported #####")
    print('',*inf.llms_supported, sep="\n\t")
    print("#"*50)
    
    int_loop(inf)