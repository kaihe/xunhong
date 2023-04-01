import json
import os

API_ERROR = 'data not found in API!!'


out_file = 'train_data/belle_metra_diago.json'
examples = []

with open('raw_data/Belle_open_source_1M.json') as fin:
    for line in fin.readlines():
        record = json.loads(line.strip())

        _input = '客户：'+record['instruction']+'\n悠悠：'
        _output = record['output']

        examples.append({
            "input":_input,
            "output":_output
        })

with open('raw_data/metra_aplaca_10000.json') as fin:
    for record in json.load(fin):
        examples.append({
            "input":record['input'],
            "output":record['output']
        })


with open('raw_data/metra_aplaca_faq.json') as fin:
    for record in json.load(fin):
        examples.append({
            "input":record['input'],
            "output":record['output']
        })

print(len(examples))

with open(out_file, 'w+') as fout:
    json.dump(examples, fout, ensure_ascii=False)