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
from chain_bots.chatglm_llm import ChatGLM, LlamaModel, BloomModel
import torch

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

    llm=OpenAI(temperature=0, max_tokens=512)

    agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory, agent_kwargs={'prefix':PREFIX, 'suffix':SUFFIX, 'format_instructions':FORMAT_INSTRUCTIONS})

    return agent_chain

if __name__ == '__main__':
    agent_chain = get_bot()
    # agent_chain.run(input="汉溪长隆到机场南的首班车是几点？")
    # agent_chain.run(input="车票多少钱？")
    # agent_chain.run(input="再列下换乘路线")
    # agent_chain.run(input="华新汇附近的地铁站是哪个？")
    import gradio as gr

    chatbot = gr.Chatbot().style(color_map=("green", "pink"))
    demo = gr.Interface(
        fn=agent_chain.run,
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
