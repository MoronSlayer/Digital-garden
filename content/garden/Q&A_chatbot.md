---
title: "Q&A chatbot with OpenAI Langchain and ChromaDB"
date: 2024-02-16
lastmod: 2024-02-16
draft: false
garden_tags: ["large-language-models"]
summary: "quick notes on dev iterations"
status: "seeding"
---

A baisc version of a Q&A chatbot using Chroma DB as the vector database and the OpenAI api is shown below:

```python
!pip -q install langchain openai tiktoken chromadb 
!wget -q https://www.dropbox.com/s/vs6ocyvpzzncvwh/new_articles.zip
!unzip -q new_articles.zip -d new_articles
import os

os.environ["OPENAI_API_KEY"] = "YOUR_KEY"
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader('./new_articles/', glob="./*.txt", loader_cls=TextLoader)

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

persist_directory = 'db'

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)

vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

retriever = vectordb.as_retriever()

docs = retriever.get_relevant_documents("How much money did Pando raise?")

retriever = vectordb.as_retriever(search_kwargs={"k": 2})


qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)
'''
function to cite sources
'''
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

# some examples:
query = "How much money did Pando raise?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "What is generative ai?"
llm_response = qa_chain(query)
process_llm_response(llm_response)
'''
Now to use the vector DB as without having to recreate the emdeddings for documents,
we can use the below code
'''
!zip -r db.zip ./db

# To cleanup, you can delete the collection
vectordb.delete_collection()
vectordb.persist()

# delete the directory
!rm -rf db/
'''
Now you can delete your runtime after you save your zipped vector db somewhere locally. To re-use your embeddings, follow the next steps 
'''
!unzip db.zip
import os

os.environ["OPENAI_API_KEY"] = "YOUR_KEY"
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

persist_directory = 'db'
embedding = OpenAIEmbeddings()

vectordb2 = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding,
                   )

retriever = vectordb2.as_retriever(search_kwargs={"k": 2})

# Set up the turbo LLM, for a more chat-like experience
turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)
# create the chain to answer questions 
qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)
## Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
query = "How much money did Pando raise?"
llm_response = qa_chain(query)
process_llm_response(llm_response)                               

```

Now, to give it a more chat like experience for a user, 

```python 
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)
```
should be replaced as this does only basic retrieval, using something like ConversationalRetrievalChain suits it much better as here history is retained and chain is more conversational.
```python

from langchain.chains import ConversationalRetrievalChain

conv_chain = ConversationalRetrievalChain.from_llm(llm=OpenAI(), 
                                                retriever=retriever,
                                                return_source_documents= True)

query = "How much money did Pando raise?"
result = conv_chain({"question": query, "chat_history": []})
answer = result["answer"]
print(answer)

followup_query = "When was their last funding round?"
result = conv_chain({"question": followup_query, "chat_history": [(query, answer)]})
followup_answer = result["answer"]
print(followup_answer)

```

But here too, we have to manually give chat history as input. To overcome this tedious process,  LangChain provides some convenient built-in tools to automate chat history management in conversational chains:

```python
'''
using RunnableWithMessageHistory -  to automatically maintain chat history between calls
'''
from langchain_core.runnables.history import RunnableWithMessageHistory

conv_chain_with_history = RunnableWithMessageHistory(
    conv_chain, 
    chat_message_history # your chat history instance
)

result = conv_chain_with_history({"input": first_query})
result = conv_chain_with_history({"input": followup_query}) 
```
OR 
```python

'''
Using a ChatMessageHistory instance - The history instance stores chat turns and can
be passed directly to the chain
'''

from langchain.memory import ChatMessageHistory

history = ChatMessageHistory() 

first_result = conv_chain({"question": first_query, "chat_history": history.messages})
history.add_conversation_turn(first_query, first_result["answer"])

second_result = conv_chain({"question": second_query, "chat_history": history.messages})
# Further turns...
```

Now, I want to use RunnableWithMessageHistory, but I am currently not depending on any external storage for my message histroy. To solve this, I am going to use an in-memory ChatMessageHistory instance and pass it to RunnableWithMessageHistory