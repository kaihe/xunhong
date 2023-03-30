import json
import os

API_ERROR = 'data not found in API!!'

files = ['sample/merge.json','sample/metra_aplaca.json','sample/metra_aplaca_faq.json']
out_file = 'sample/metra_5000_merge.json'

examples = []

for f in files:
    if 'metra' in f:
        data = json.load(open(f))
        data = [d for d in data if API_ERROR not in d['input']]
        examples.extend(data)

    examples.extend(json.load(open(f)))

with open(out_file, 'w+') as fout:
    json.dump(examples, fout)