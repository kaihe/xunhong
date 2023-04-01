import json
import tqdm
from template import generate_prompt
from transformers import LlamaForCausalLM, LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf', add_eos_token=True)

def his_len():

    data = json.load(open(r'raw_data\metra_aplaca_10000.json'))
    prompt_size = []
    for d in tqdm.tqdm(data):
        result = tokenizer(generate_prompt(d['input'], d['output']),truncation=False)
        prompt_size.append(len(result['input_ids']))

    import matplotlib.pyplot as plt

    plt.hist(prompt_size, bins=20)
    plt.show()


def peek_data():
    import re
    examples = json.load(open(r'sample\merge.json'))
    # for d in examples:
    #     if re.match('.*[\:|：].*', d['input']):
    #         print(d)
    print(len([d for d in examples if d['input']==""])) 

if __name__ == '__main__':
    his_len()