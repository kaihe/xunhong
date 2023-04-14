# chatllm parse api call inputs
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.utilities import GoogleSearchAPIWrapper
import openai
import os
from dotenv import load_dotenv
load_dotenv("./.env")
from chain_bots.metra_tools import mtools

search = GoogleSearchAPIWrapper()

llm = OpenAI(temperature=0)
tools = mtools + [Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    )]
mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

mrkl.run("怎么乘地铁去体育西路站?")