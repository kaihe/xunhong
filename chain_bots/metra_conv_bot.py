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
