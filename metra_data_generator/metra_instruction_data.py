from metra_api import MetraData, API_ERROR, CALL_API
import pandas as pd
import random
import copy, tqdm
from metra_data_generator.conv_template import gen_template_data

metra = MetraData(reload=False)

INSTRUCTION="""扮演广州地铁机器人悠悠，把客户问题转化为API调用指令"""

template = {
    "instruction":INSTRUCTION,
    "input":"",
    "output":""
}


def make_faq_data():
    f_path = 'data/raw/智能客服知识库20221213.xls'
    sheet_dict = pd.read_excel(f_path, sheet_name=None)

    examples = []

    for tab, df in sheet_dict.items():
        for q, a in zip(df['问题'], df['机器人回复']):

            example = copy.deepcopy(template)
            example['input'] = '客户：'+q+ "\n悠悠："
            example['output'] = a
            examples.append(example)
    
    return examples

def make_metra_data(count=5000):
    convs = gen_template_data(count)
    convs = [text.replace('<<CALL_API>>', CALL_API) for text in convs]

    # randomly split conv for multi round ones
    examples = []
    for conv in tqdm.tqdm(convs):
        lines = conv.split('\n')
        if len(lines)>2:
            cut = random.choice(range(1, len(lines), 2))+1
            lines = lines[:cut]

        assert len(lines)>1 and len(lines) % 2 == 0

        # if there is api call in _input, proceed to that api call for real bot answer
        _proceed_lines = [metra.proceed_api_call(s) for s in lines[:-1]]
        _input = '\n'.join(_proceed_lines)+ "\n悠悠："

        if API_ERROR in _input:
            continue

        _output = lines[-1].replace('悠悠：','')
        example = copy.deepcopy(template)
        example['input'] = _input
        example['output'] = _output
        examples.append(example)
    
    return examples


if __name__ == '__main__':
    import json
    cnt = 5000
    data = make_metra_data(cnt)
    print(f'writing {len(data)} records')
    out=f'raw_data/metra_aplaca_{cnt}.json'
    with open(out, 'w+') as fout:
        json.dump(data, fout, ensure_ascii=False)

    # import json
    # data = json.load(open(r'data\tmp\metra_aplaca_faq.json'))
    # for d in data:
    #     d['instruction'] = INSTRUCTION
    # with open(r'data\tmp\metra_aplaca_faq.json','w+') as fout:
    #     json.dump(data, fout, ensure_ascii=False)


    # data = make_faq_data()
    # print(f'writing {len(data)} records')
    # out=r'data\tmp\metra_aplaca_faq.json'
    # with open(out, 'w+') as fout:
    #     json.dump(data, fout, ensure_ascii=False)