from langchain.llms.base import LLM
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy, BloomForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE_ID = "0" if torch.cuda.is_available() else None
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

class BloomModel(LLM):
    tokenizer: object = None
    model: object = None
    max_new_tokens:int = 512

    def __init__(self, model_name='bigscience/bloom-1b1'):
        super().__init__()
        self.tokenizer= AutoTokenizer.from_pretrained(model_name)
        self.model  = AutoModelForCausalLM.from_pretrained(model_name, use_cache=True).cuda()

    @property
    def _llm_type(self) -> str:
        return "bloom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to('cuda')

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=512
            )

        output = self.tokenizer.decode(generation_output[0])

        torch_gc()
        if stop is not None:
            response = enforce_stop_tokens(output, stop)

        return response

class LlamaModel(LLM):
    tokenizer: object = None
    model: object = None
    max_new_tokens:int = 512

    def __init__(self, model_name='decapoda-research/llama-7b-hf'):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map='auto'
        ).cuda().eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

    @property
    def _llm_type(self) -> str:
        return "llama"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to('cuda')

        generation_output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens
        )

        s = generation_output.sequences[0]
        response = self.tokenizer.decode(s)

        torch_gc()
        if stop is not None:
            response = enforce_stop_tokens(response, stop)

        return response

class ChatGLM(LLM):
    max_token: int = 100000
    temperature: float = 0.01
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None
    history_len: int = 10

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history[-self.history_len:] if self.history_len>0 else [],
            max_length=self.max_token,
            temperature=self.temperature,
        )
        torch_gc()
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history+[[None, response]]
        return response

    def load_model(self, model_name_or_path: str = "THUDM/chatglm-6b"):
        if model_name_or_path == 'chatglm':
        
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )
            
            self.model = (
                AutoModel.from_pretrained("THUDM/chatglm-6b-int4",trust_remote_code=True)
                .half().cuda()
            )
        elif model_name_or_path == 'llama':
            model_name = 'decapoda-research/llama-7b-hf'
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name,add_eos_token=True)
            self.tokenizer.pad_token_id = 0
            self.model = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")

        self.model = self.model.eval()