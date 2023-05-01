from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain import OpenAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent
from dotenv import load_dotenv
load_dotenv("./.env")
from chain_bots.metra_tools import mtools
from chain_bots.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.utilities import GoogleSearchAPIWrapper
from chain_bots.chatglm_llm import ChatGLM, LlamaModel, BloomModel
import torch
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from chain_bots.doc_search import get_metro_search_tool

class BotManager:
    bots = {}
    def __init__(self) -> None:
        pass
    
    def get_bot(self, user_id):
        if user_id in self.bots:
            return self.bots[user_id]
        
        else:
            return self.init_bot(user_id)
    
    def init_bot(self, user_id):

        llm=OpenAI(temperature=0, max_tokens=512)
        search = GoogleSearchAPIWrapper()
        metro_search = get_metro_search_tool()

        tools = [
            Tool(
                name = "Google Search",
                func=search.run,
                description="you need this tool to provide fact based answers, input is keywords in Chinese"
            ),
            Tool(
                name = "Metro Search",
                func=metro_search.run,
                description="you need this tool to answer Guangzhou Metro FAQ questions, input the query"
            )
        ] + mtools

        memory = ConversationBufferWindowMemory(k=4, memory_key="chat_history")
        
        agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory, agent_kwargs={'prefix':PREFIX, 'suffix':SUFFIX, 'format_instructions':FORMAT_INSTRUCTIONS})

        self.bots[user_id] = agent_chain

        return agent_chain

def demo():
    manager = BotManager()

    import gradio as gr
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        user_id = gr.Textbox(placeholder='请输入一个用户名')
        msg = gr.Textbox(placeholder='在这里输入内容，开始聊天吧')

        clear = gr.Button("Clear")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(user_name, history):
            agent_chain = manager.get_bot(user_name)
            bot_message = agent_chain.run(input=history[-1][0])
            history[-1][1] = bot_message
            return history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [user_id, chatbot], chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch(share=True)

def test():
    manager = BotManager()
    agent = manager.get_bot('test')

    # agent.run('从机场南到大学城南要怎么换乘，票价多少钱')
    # agent.run('我只有8块钱，能从机场南坐车到大学城南吗？')
    # agent.run('从机场南到大学城南坐地铁要多久？')
    agent.run('广州地铁车票能退吗？')
if __name__ == '__main__':
    demo()
    # test()
