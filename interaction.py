import sys
import torch
from peft import PeftModel
import transformers
import gradio as gr
import argparse
import warnings
import os
from metra_api import MetraData
metra_api = MetraData(reload=False)

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
parser.add_argument("--lora_path", type=str, default="./lora-Vicuna/checkpoint-final")
parser.add_argument("--use_local", type=int, default=1)
args = parser.parse_args()

tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

LOAD_8BIT = True
BASE_MODEL = args.model_path
LORA_WEIGHTS = args.lora_path

if not os.path.exists(os.path.join(args.lora_path, 'adapter_config.json')):
    import shutil
    shutil.copyfile('config-sample/adapter_config.json', os.path.join(args.lora_path, 'adapter_config.json'))

# fix the path for local checkpoint
lora_bin_path = os.path.join(args.lora_path, "adapter_model.bin")
print(lora_bin_path)
if not os.path.exists(lora_bin_path) and args.use_local:
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

def interaction(
    input,
    history,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    repetition_penalty=1.0,
    max_memory=256,
    **kwargs,
):
    now_input = input
    history = history or []
    if len(history) != 0:
        input = "\n".join(["客户：" + i[0]+"\n"+"悠悠：" + i[1] for i in history]) + "\n" + input + "\n" + "悠悠："
        if len(input) > max_memory:
            input = input[-max_memory:]

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

    bot_line = output.split('\n')[-1]
    bot_ans = metra_api.proceed_api_call(bot_line)
    history.append(('客户：'+now_input, "悠悠："+bot_ans))
    print('客户：'+now_input)
    print('debug: '+bot_line)
    print(bot_ans)
    print('-----')
    return history, history

chatbot = gr.Chatbot().style(color_map=("green", "pink"))
demo = gr.Interface(
    fn=interaction,
    inputs=[
        gr.components.Textbox(
            lines=2, label="Input", placeholder="Tell me about alpacas."
        ),
        "state",
        gr.components.Slider(minimum=0, maximum=1, value=1.0, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.9, label="Top p"),
        gr.components.Slider(minimum=0, maximum=100, step=1, value=60, label="Top k"),
        gr.components.Slider(minimum=1, maximum=5, step=1, value=2, label="Beams"),
        gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=128, label="Max new tokens"
        ),
        gr.components.Slider(
            minimum=0.1, maximum=10.0, step=0.1, value=2.0, label="Repetition Penalty"
        ),
        gr.components.Slider(
            minimum=0, maximum=2000, step=1, value=256, label="max memory"
        ),
    ],
    outputs=[chatbot, "state"],
    allow_flagging="auto",
    title="广州地铁机器人",
    description="可以进行开放领域问答，可以根据外部API进行广州地铁换乘查询，车票查询，设施查询，车站时间查询等",
)
demo.queue().launch(share=True, inbrowser=True)