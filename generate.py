import sys
import torch
from peft import PeftModel
import transformers
# import gradio as gr
import argparse
import warnings
import os
from template import generate_prompt
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from metra_api import MetraData
metra_api = MetraData(reload=False)


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
parser.add_argument("--lora_path", type=str, default="./lora-Vicuna/checkpoint-3800")
args = parser.parse_args()

tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

LOAD_8BIT = True
BASE_MODEL = args.model_path
LORA_WEIGHTS = args.lora_path

if not os.path.exists(os.path.join(args.lora_path, 'adapter_config.json')):
    import shutil
    shutil.copyfile('config-sample/adapter_config.json', os.path.join(args.lora_path, 'adapter_config.json'))

lora_bin_path = os.path.join(args.lora_path, "adapter_model.bin")
print(lora_bin_path)
if not os.path.exists(lora_bin_path):
    pytorch_bin_path = os.path.join(args.lora_path, "pytorch_model.bin")
    print(pytorch_bin_path)
    if os.path.exists(pytorch_bin_path):
        os.rename(pytorch_bin_path, lora_bin_path)
        warnings.warn("The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'")
    else:
        assert ('Checkpoint is not Found!')
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        torch_dtype=torch.float16,
        device_map={'': 0}
    )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )

if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(
    input,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=2,
    max_new_tokens=128,
    repetition_penalty=1.0,
    **kwargs,
):
    inputs = tokenizer(input, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            repetition_penalty=float(repetition_penalty),
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)

    bot_line = output.split('悠悠：')[-1]

    output = metra_api.proceed_api_call(bot_line)
    return output



if __name__ == "__main__":
    # testing code for readme
    for input in [
        "客户：用一句话描述地球为什么是独一无二的。\n悠悠：", 
        "客户：红楼梦后四十回是曹雪芹写的吗？为什么\n悠悠：",
        "客户：1996是闰年吗。\n悠悠：", 
        "客户：列出广州地铁三号线的车站\n悠悠：",
        "客户：汉溪长隆站最早几点钟能有车？\n悠悠：",
        "客户：同济路去平南的地铁车票要花多少钱？\n悠悠：从同济路到平南的地铁票价是6元\n客户：目的地改为燕塘\n悠悠：",
        "客户：分别列出沥滘的第一班车和最后一班车时间\n悠悠：",
        "客户：往新丰路的呢？\n悠悠：从峻泰路到新丰路的地铁票价是2元\n客户：首班车和末班车分别是几点？\n悠悠：峻泰路前往新丰路的首班车时间是6:32。，峻泰路前往新丰路的末班车时间是22:48。\n客户：帮我找下出发地的洗手间\n悠悠：",
        "客户：钟落潭站的母婴室在哪里？\n悠悠：",
        "客户：中国篮球第一人是谁？\n悠悠：",
        "客户：你觉得香港的国安法怎么样？\n悠悠：",
        "客户：江泽民是个好官吗？\n悠悠：",   
        "客户：列10个许姓女宝宝的名字\n悠悠：",   
        "客户：鼻炎要怎么治疗？\n悠悠：",      
        "客户：列一下周杰伦的新歌\n悠悠：",   
        "客户：对对联，南通州北通州南北通州通南北\n悠悠：",      
        "客户：人生的意义是什么？\n悠悠：",   
        "客户：什么是二律背反？\n悠悠：",         
    ]:
        print(input+evaluate(input))
        print('-------------')




