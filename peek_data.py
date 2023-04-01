import json
import tqdm
# from transformers import LlamaForCausalLM, LlamaTokenizer
# tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf', add_eos_token=True)

def post_process_data():

    def generate_prompt(data_point):
        # sorry about the formatting disaster gotta move fast
        if data_point["input"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Input:
    {data_point["input"]}

    ### Response:
    {data_point["output"]}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Response:
    {data_point["output"]}"""


    _new_instruct = "扮演广州地铁机器人悠悠，把客户问题转化为API调用指令"

    data = json.load(open(r'sample\metra_aplaca_faq.json'))
    prompt_size = []
    for d in tqdm.tqdm(data):
        d['instruction'] = _new_instruct
        _input = d['input']
        _input_lines = _input.split('\n')
        if len(_input_lines)>6:
            _input_lines = _input_lines[-6:]
            d['input'] = '\n'.join(_input_lines)

        # result = tokenizer(generate_prompt(d),truncation=False)
        # prompt_size.append(len(result['input_ids']))

    # import matplotlib.pyplot as plt

    # plt.hist(prompt_size, bins=20)
    # plt.show()

    with open(r'sample\metra_aplaca_faq.json', 'w+') as fout:
        json.dump(data, fout, ensure_ascii=False)

def peek_data():
    import re
    examples = json.load(open(r'sample\merge.json'))
    # for d in examples:
    #     if re.match('.*[\:|：].*', d['input']):
    #         print(d)
    print(len([d for d in examples if d['input']==""])) 

if __name__ == '__main__':
    peek_data()