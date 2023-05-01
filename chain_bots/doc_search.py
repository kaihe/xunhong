from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.tools import BaseTool
from dotenv import load_dotenv
load_dotenv("./.env")

def gen_doc():
    import json
    f_path = r'raw_data\metra_aplaca_faq.json'
    data = json.load(open(f_path))
    faqs = [d['input']+d['output'] for d in data]
    
    faqs = [d.replace('客户：','Q:').replace('悠悠：','A:') for d in faqs]
    
    with open(r'raw_data\faq_doc.txt','w+') as fout:
        for d in faqs:
            fout.write(d+'\n')
            fout.write('----\n')

def get_metro_search_tool():
    from langchain.chains import RetrievalQA
    from langchain import OpenAI
    # name = "Metra search"
    # description = "useful for when you need to answer faq questions about Guangzhou Metro"
    
    with open(r"raw_data\faq_doc.txt") as f:
        state_of_the_union = f.read()
    text_splitter = CharacterTextSplitter(separator='----',chunk_size=1,chunk_overlap=0)
    texts = text_splitter.split_text(state_of_the_union)

    embeddings = OpenAIEmbeddings()

    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever(search_type="similarity", search_kwargs={"k":2})

    qa_tool = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=docsearch, return_source_documents=False)

    return qa_tool

def test():

    with open(r"raw_data\faq_doc.txt") as f:
        state_of_the_union = f.read()
    text_splitter = CharacterTextSplitter(separator='----',chunk_size=1,chunk_overlap=0)
    texts = text_splitter.split_text(state_of_the_union)

    embeddings = OpenAIEmbeddings()

    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()

    query = "地铁车票怎么退"
    docs = docsearch.get_relevant_documents(query)

    print(docs)

if __name__ == '__main__':
    # gen_doc()
    # test()
    qa_tool = get_metro_search_tool()
    print(qa_tool('为什么要安检？'))