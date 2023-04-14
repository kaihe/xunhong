import openai
import os
from dotenv import load_dotenv
load_dotenv("./.env")
openai.api_key = os.getenv("OPENAI_API_KEY")


def test_csv():
    from langchain.agents import create_csv_agent
    from langchain.llms import OpenAI
    # openai_api_key='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'
    llm = OpenAI(temperature=0)
    agent = create_csv_agent(llm, 'data/titanic.csv', verbose=True)

    agent.run("男乘客有几位？")

def test_api_call():
    from typing import List, Optional
    from langchain.chains import LLMChain
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.requests import Requests
    from langchain.tools import APIOperation, OpenAPISpec
    from langchain.agents import AgentType, Tool, initialize_agent
    from langchain.agents.agent_toolkits import NLAToolkit

    # Select the LLM to use. Here, we use text-davinci-003
    llm = OpenAI(temperature=0, max_tokens=700) # You can swap between different core LLM's here.

    speak_toolkit = NLAToolkit.from_llm_and_url(llm, "https://api.speak.com/openapi.yaml")
    klarna_toolkit = NLAToolkit.from_llm_and_url(llm, "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/")

    # Slightly tweak the instructions from the default agent
    openapi_format_instructions = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: what to instruct the AI Action representative.
Observation: The Agent's response
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer. User can't see any of my observations, API responses, links, or tools.
Final Answer: the final answer to the original input question with the right amount of detail

When responding with your Final Answer, remember that the person you are responding to CANNOT see any of your Thought/Action/Action Input/Observations, so if there is any relevant information there you need to include it explicitly in your response."""

    natural_language_tools = speak_toolkit.get_tools() + klarna_toolkit.get_tools()
    mrkl = initialize_agent(natural_language_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                            verbose=True, agent_kwargs={"format_instructions":openapi_format_instructions})
    
    mrkl.run("I need you to explain some Chinese sentence to me, 君子之交淡如水")

def test_custom_tool():
    # chatllm parse api call inputs
    from langchain.llms import OpenAI
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType
    from langchain.utilities import GoogleSearchAPIWrapper
    
    search = GoogleSearchAPIWrapper()

    def query_train_time(string):
        departure, arrival = string.split(",")
        if departure == "None":
            return "请问从哪个车站出发？"
        elif arrival == "None":
            return "请问去哪个车站？"
        elif departure == "None" and arrival == "None":
            return "请问从哪个站出发，到哪个站？"
        else:
            return f"从{departure}到{arrival}的首班车是6:10，末班车是23:10"

    def query_route(string):
        departure, arrival = string.split(",")
        if departure == "None":
            return "请问从哪个车站出发？"
        if arrival == "None":
            return "请问去哪个车站？"
        elif departure == "None" and arrival == "None":
            return "请问从哪个站出发，到哪个站？"
        else:
            return f"从{departure}到{arrival}的要先乘坐3号线到体育西路，再换乘7号线"

    def query_stations(string):
        return f'{string}的车站有厦滘，沥滘，汉溪长隆，南村万博，大学城南'

    def query_device(string):
        station, device_type = string.split(",")
        if station == "None":
            return "请问您在哪个车站？"
        elif device_type == "None":
            return "请问要查询那种设备？"
        elif station == "None" and device_type == "None":
            return "请问您要查询哪个站的什么设备？"
        return f'{station}的{device_type}在A出口'

    def ask_for_information(string):
        return string

    llm = OpenAI(temperature=0)
    tools = [
        Tool(
            name = "Train time query",
            func=query_train_time,
            description="useful for when you need to answer train departure and arrival time. The input to this tool should be a comma separated list of numbers of length two, representing the departure station and arrival station. Put None if station is unclear. For example, `汉溪长隆,南村万博` would be the input if depart from 汉溪长隆 station and arrive at 南村万博 station. `汉溪长隆,None` would be the input if depart from 南村万博 and the arrival station is unknown.`None,None` would be the input if user haven't specified the departure and arrival station"
        ),
        Tool(
            name = "Transfer query",
            func=query_route,
            description="useful for when you need to answer train route transfer information. The input to this tool should be a comma separated list of numbers of length two, representing the departure station and arrival station. Put None if station is unclear. For example, `汉溪长隆,南村万博` would be the input if depart from 汉溪长隆 station and arrive at 南村万博 station. `汉溪长隆,None` would be the input if depart from 南村万博 and the arrival station is unknown.`None,None` would be the input if user haven't specified the departure and arrival station"
        ),
        Tool(
            name = "List stations",
            func=query_stations,
            description="useful for when you need to list stations of a specific line. Input is the line name. For example, `三号线`,`APM线`,`十三号线`. `None` if user haven't specify which line."
        ),
        Tool(
            name = "Device query",
            func=query_device,
            description="useful for when you need to find a specific device in a station. The input to this tool should be a comma separated list of numbers of length two, representing the station name and device name. Put None if any of them is unclear. For example, `汉溪长隆,卫生间` would be the input if you want to find device 卫生间 in 汉溪长隆 station. `None,面包店` would be the input if you want to find 面包店 but the station is unknown. `None,None` would be the input if the user haven't specify both the station name and device name yet."
        ),
        Tool(
            name = "Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world"
        ),
        Tool(
            name = "Information specifier",
            func = ask_for_information,
            description="use this tool if you need to ask user to specify his/her query"
        )
    ]
    mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    mrkl.run("怎么去西门口?")

def test_final_bot():
    from langchain.agents import Tool
    from langchain.agents import AgentType
    from langchain.memory import ConversationBufferMemory
    from chain_bots.test_chain import OpenAI
    from langchain.utilities import GoogleSearchAPIWrapper
    from langchain.agents import initialize_agent
    
    search = GoogleSearchAPIWrapper()
    tools = [
        Tool(
            name = "Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world"
        ),
    ]

    memory = ConversationBufferMemory(memory_key="chat_history")
    llm=OpenAI(temperature=0)
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
    agent_chain.run(input="hi, i am bob")

if __name__ == '__main__':
    test_custom_tool()