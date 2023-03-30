import json
import random

data = json.load(open(r'sample\merge.json'))
samples = random.sample(data, 5120)

with open(r'sample/5120_samples.json','w+') as fout:
    json.dump(samples, fout, ensure_ascii=False)