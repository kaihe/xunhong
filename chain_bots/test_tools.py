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
from langchain import LLMMathChain


search = GoogleSearchAPIWrapper()

llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

tools = [Tool(
            name = "Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world. Don't input complex search keyword, split complex query into multiple simple ones and use this tool multiple times."
         ),Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math like agg difference"
        )]
PREFIX = """Answer the following questions as best you can. Your final answer should in Chinese. You have access to the following tools:"""
mrkl = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True,agent_kwargs={'prefix':PREFIX})

mrkl.run("刘德华的年龄比姚明大多少岁？")