from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent
from dotenv import load_dotenv
load_dotenv("./.env")
from chain_bots.metra_tools import mtools
from chain_bots.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.utilities import GoogleSearchAPIWrapper
from chatglm_llm import ChatGLM
import torch

wikipedia = WikipediaAPIWrapper()
search = GoogleSearchAPIWrapper()


from langchain.llms.base import LLM
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE_ID = "0" if torch.cuda.is_available() else None
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class ChatGLM(LLM):
    max_token: int = 512
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        if torch.cuda.is_available():
            self.model = (
                AutoModel.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True)
                .half()
                .cuda()
            )
        elif torch.backends.mps.is_available():
            self.model = (
                AutoModel.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True)
                .float()
                .to('mps')
            )
        else:
            self.model = (
                AutoModel.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True)
                .float()
            )
        self.model = self.model.eval()

def get_bot():

    tools = [
        Tool(
            name = "Google Search Tool",
            func=search.run,
            description="always use this tool first before giving answer about factual questions"
        )
    ] + mtools

    memory = ConversationBufferMemory(memory_key="chat_history")

    # llm=OpenAI(temperature=0, max_tokens=512)

    llm = ChatGLM()
    llm.load_model()


    agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory, agent_kwargs={'prefix':PREFIX, 'suffix':SUFFIX, 'format_instructions':FORMAT_INSTRUCTIONS})

    return agent_chain

if __name__ == '__main__':
    agent_chain = get_bot()

    agent_chain.run(input="汉溪长隆到机场南的首班车是几点？")
    agent_chain.run(input="车票多少钱？")
    agent_chain.run(input="再列下换乘路线")
    agent_chain.run(input="华新汇附近的地铁站是哪个？")
