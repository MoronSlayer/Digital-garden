---
title: "RAG: Chat with data"
date: 2024-02-16
lastmod: 2024-02-17
draft: false
garden_tags: ["large-language-models"]
summary: "RAG with LangChain"
status: "seeding"
---


# RAG

Retrieval Augmented Generation(RAG) is a popular process to establish a connection with any local data source to large language models as shown in the diagram below.
{{< figure src="./RAG.png"  width="100%" >}}

We shall see the process of constructing these applications and go through the steps listed in the diagram. You can also code along. We will be working with the OpenAI api, but any LLM can be used.

```python
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 

openai.api_key  = os.environ['OPENAI_API_KEY']
```

## Step 1 - Document loaders

The first step to chat with any kind of data source you might have is to bring your data sources into a format that can be work with. Document loaders help with this task.

Some examples are listed

#### PDFs
[Here is a demo pdf you can work with](https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture01.pdf)
```python

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/Lecture_notes.pdf")
pages = loader.load()

len(pages)
page = pages[0]
page.metadata

```

Each page is a `Document`.

A `Document` contains text (`page_content`) and `metadata`.

#### Youtube videos

```python
! pip install yt_dlp
! pip install pydub
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir="docs/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),
    OpenAIWhisperParser()
)
docs = loader.load()
docs[0].page_content[0:500]
```

#### URLs

```python
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/37signals-is-you.md")
docs = loader.load()
print(docs[0].page_content[:500])
```

#### Notion

```python
from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()
print(docs[0].page_content[0:200])
docs[0].metadata
```

## Step 2 - Document Splitting

For large documents, they need to be split into chunks to make it more easy for the model to understand your data.

{{< figure src="./chunk_basics.png"  width="100%" >}}

As seen above, the basics in langchain involve splitting on some chunks in some chunk size with some chunk overlap.

There are different kinds of text splitters in langchain, based on various factors. They can vary on how they split the chunks, what characters go into that. They can vary on how they measure the length of the chunks. 
Is it by characters? Is it by tokens? There are even some that use other smaller models to determine when the end of a sentence might be and use that as a way of splitting chunks. Another important part of splitting into chunks is also the metadata. Maintaining the same metadata across all chunks, but also adding in new pieces of metadata when relevant, and so there are some text splitters that are really focused on that. There are also some splitters based on specific programming languages such as Python or Ruby that take into account the semantics of the language.

##### Here's an example with some popular splitters, ```RecursiveCharacterTextSplitter``` and ```CharacterTextSplitter```

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
chunk_size =26
chunk_overlap = 4
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

text1 = 'abcdefghijklmnopqrstuvwxyz'
r_splitter.split_text(text1)

text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
r_splitter.split_text(text2)

text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
r_splitter.split_text(text3)
```

We can see that text1 is not split as it is 26 letters and our chunk size is 26

```python
c_splitter.split_text(text3)
```
Now we see that the character is not split, the reason is that CharacterTextSplitter splits by characters and by deafult that means on new lines, to overcome this we can set the separator to an empty space.

```python
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator = ' '
)
c_splitter.split_text(text3)
```
This works now!

```RecursiveCharacterTextSplitter` ``` is recommended for generic text 

```python
some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator = ' '
)
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0, 
    separators=["\n\n", "\n", " ", ""]
)

c_splitter.split_text(some_text)

r_splitter.split_text(some_text)
```

You can reduce the chunk size and experiment.

We can also set complex regex in the separators for more legible chunking

```python
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)
r_splitter.split_text(some_text)
```

##### Now for documents,

```python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/Lecture.pdf")
pages = loader.load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

docs = text_splitter.split_documents(pages)
len(docs)
len(pages)
```

##### We can also split based on tokens explicitly, which are how LLMs are based on:

```python
from langchain.text_splitter import TokenTextSplitter
text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
text1 = "foo bar bazzyfoo"
text_splitter.split_text(text1)
text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
docs = text_splitter.split_documents(pages)
docs[0]
pages[0].metadata
```

##### Context-aware chunking

Chunking aims to keep text with common context together.

A text splitting often uses sentences or other delimiters to keep related text together but many documents (such as Markdown) have structure (headers) that can be explicitly used in splitting.

We can use MarkdownHeaderTextSplitter to preserve header metadata in our chunks, as show below

```python
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n 
## Chapter 2\n\n \
Hi this is Ritesh"""
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(markdown_document)
md_header_splits[0]
md_header_splits[1]
```

The header data is used for seeing where to split and it is saved in the metadata for referencing later.


## Step 3 - Vector Stores and Embeddings

Now, with semantically meaningfully split chunks of our documents, we can put them into an index so that we can retrieve them easily when we have answer questions about them.

Embeddings/vectors capture semantic meaning in our text. Text with similar meaning will have similar embeddings.

{{< figure src="./embedding_vecs.png"  width="100%" >}}


```python
from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)
len(splits)

# Now we can take our splits and embed them

from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

import numpy as np

np.dot(embedding1, embedding2)
np.dot(embedding1, embedding3)
np.dot(embedding2, embedding3)
```

We will see that the do product of sentence 1 and 2 is the highest as it corresponds to similar sentences.

In a vector store, we create embeddings our document and store them. Then when we have questions that we want to ask the documents through our LLMs,  our question to the LLM passes as embeddings back to the vector store to retrieve the 'k' most relevant documents related to our question.

{{< figure src="./create_embeddings.png"  width="100%" >}}

{{< figure src="./index_embeddings.png"  width="100%" >}}

Now, let's explore this process with an open source lightweight, in-memory vector store, Chroma.

```python

pip install chromadb

from langchain.vectorstores import Chroma

#persist directory is where our vector store will be saved which can be reused later on 
persist_directory = 'docs/chroma/'
!rm -rf ./docs/chroma  # remove old database files if any
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
print(vectordb._collection.count())

```

Now, coming to the similarity search part

```python

question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question,k=3)
len(docs)
docs[0].page_content

# to save this for later use 
vectordb.persist()

```

Although this is great, and similarity search works well, there can be edge cases that creep up.
Suppose we have a query such as 

```python

question = "what did they say about matlab?"
docs = vectordb.similarity_search(question,k=5)
```

This query might return the same chunks in the search as they might contain the same documents twice or more. Because the same information lies in two different chunks, we get issues. Semantic search fetches all similar documents, but does not enforce diversity. It would be much better if there if there is a different, distinct chunk that our model can learn from.

Another type of failure is seen below

```python
question = "what did they say about regression in the third lecture?"
docs = vectordb.similarity_search(question,k=5)
for doc in docs:
    print(doc.metadata)
```
Here the question is from lecture three but we also get other lectures in our results which don't contain information about regression. We will adress these issues next


## Step 4 - Retrieval

Retrieval is the centerpiece of our retrieval augmented generation (RAG) flow.

Below is the common code used to load our vector store for all tasks in retrieval.

```python
pip install lark

##  here, we load the vector database just as we did previously

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print(vectordb._collection.count())

```

Some of the more advanced methods for retrieval include:

#### **Maximal marginal retrieval**

- You may not always want to choose most similar responses, so that you don't lose diverse information.

for e.g: A chef asks about white mushrooms that are fruiting, it results two results about Amanita phalloides. But the third result contains information that this mushroom is actually poisonous, but is not retrieved as it is not as relevant to the query the chef asked.

MMR comes into play here, where divergence comes into play.

{{< figure src="./MMR.png"  width="100%" >}}

```python
texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]

smalldb = Chroma.from_texts(texts, embedding=embedding)
question = "Tell me about all-white mushrooms with large fruiting bodies"
smalldb.similarity_search(question, k=2)
smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3)
```
Also, in one of the previous examples, we saw how our model returned multiple irrelevant docs when asked about regression. Now, we can use MMR. 

```python
question = "what did they say about matlab?"
docs_ss = vectordb.similarity_search(question,k=3)
docs_ss[0].page_content[:100]
docs_ss[1].page_content[:100]

# Now, see the differnce with MMR

docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)
docs_mmr[0].page_content[:100]
docs_mmr[1].page_content[:100]
```

#### **LLM aided retrieval/ SelfQuery**

This is useful when we have questions that aren't related just to the questions that we ask semantically but also have some reference to the metadata that we want to do a filter on.

{{< figure src="./llm_aided.png"  width="100%" >}}

Here we have the semantic context of aliens and metadata for 1980. What we can do is an LLM itself to split the question into two parts, a metadata filter and a search term.

In a previous example, we saw that a question about the third lecture can include results from other lectures as well. To address this, many vectorstores support operations on metadata. Metadata provides context for each embedded chunk.

```python
question = "what did they say about regression in the third lecture?"
docs = vectordb.similarity_search(
    question,
    k=3,
    filter={"source":"docs/cs229_lectures/MachineLearning-Lecture03.pdf"}
)
for d in docs:
    print(d.metadata)
```

**Addressing Specificity: working with metadata using self-query retriever**

But we have an interesting challenge: we often want to infer the metadata from the query itself, not do it manually.

To address this, we can use `SelfQueryRetriever`, which uses an LLM to extract:
 
1. The `query` string to use for vector search
2. A metadata filter to pass in as well

Most vector databases support metadata filters, so this doesn't require any new databases or indexes.

```python
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]

document_content_description = "Lecture notes"
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)

question = "what did they say about regression in the third lecture?"

docs = retriever.get_relevant_documents(question)

for d in docs:
    print(d.metadata)

```
The metadata_info is passed onto our LLMs, so it is important to be as specific as possible.
#### **Compression**
Compression is useful to pull only the most relevant splits of the retrieved package.
Information most relevant to a query may be buried in a document with a lot of irrelevant text. 
Passing that full document through your application can lead to more expensive LLM calls and poorer responses.
Contextual compression is meant to fix this. 


{{< figure src="./compression.png"  width="100%" >}}
As seen in the image above, our intial question might return the entire documents that were stored even though we need only one or two documents. With compression, we can run all those documents into a compression LLM and get only the most relevant documents passed to the final LLM. This comes at a cost of more calls to the LLM but it is useful for getting only the most relevant parts.

```python

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

# Wrap our vectorstore
llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)

question = "what did they say about matlab?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)

```

In the example above, we will see two things. First, the documents are a lot shorter, which is good but there also repeated documents because under the hood we still use semantic search. This can be solved using MMR.

To solve this, we can combine these two methods to get best results

```python

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type = "mmr")
)

question = "what did they say about matlab?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)

```

Now we get a filtered set of results without any duplicate information.


There are some other types of retrieval in langchain that don't use vector databases at all, such as TF-IDF or SVM which are more traditional NLP techniques.

```python
from langchain.retrievers import SVMRetriever
from langchain.retrievers import TFIDFRetriever
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
all_page_text=[p.page_content for p in pages]
joined_page_text=" ".join(all_page_text)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)
splits = text_splitter.split_text(joined_page_text)

# Retrieve
svm_retriever = SVMRetriever.from_texts(splits,embedding)
tfidf_retriever = TFIDFRetriever.from_texts(splits)

question = "What are major topics for this class?"
docs_svm=svm_retriever.get_relevant_documents(question)
docs_svm[0]

question = "what did they say about matlab?"
docs_tfidf=tfidf_retriever.get_relevant_documents(question)
docs_tfidf[0]

```

## Step 5 - Question Answering

Now, after we retrieve documents, potentially compressing the relevant chunks to fit into the LLM context; we send the information along with our question to an LLM to select and format an answer.

We can see the general flow of an information retrieval question answering system below:

{{< figure src="./retrieval_qa.png"  width="100%" >}}

We get a question, retrieve the most relevant documents in splits and send that as the system prompt along with the original human prompt to the LLM to get an answer. 

By default we pass all the chunks into the same context window, into the same call with the language model.

Some of the advantages are offered in situations where all of the documents simply can't fit inside the context window. The above listed techniques are useful to get around this issue of short context windows.

Let's look at the basic process of building a retrieval QA system with code.
```python

# We load the vector database that we created earlier

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

'''
If you have been using the same data, you can see that the collection count always 
remains constant as we are not adding any additional documents to our vector store.
'''

print(vectordb._collection.count())

question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# Now the RetrievalQA chain

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

result = qa_chain({"query": question})

result["result"]
```
To get some insights into what's happening under the hood, and understand which knobs we can turn.

The main part that's important here is the prompt, which will take in the documents and the question and pass it to the LLM.

```python   
from langchain.prompts import PromptTemplate

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = "Is probability a class topic?"
result = qa_chain({"query": question})
result["result"]
result["source_documents"][0]

```

This is the vanilla way of doing things, stuff all the documents  into the final prompt.

But if there are too many documents, they may not be able to fit into the context window.

#### **Map Reduce**

{{< figure src="./retrieval_qa_map_reduce.png"  width="100%" >}}


```python
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)
result = qa_chain_mr({"query": question})   
result["result"]
```

In this technique, each of the document is sent to the LLM to get an original answer. Then all induvidual anaswers are re-sent to the LLM to get a final call as the  final answer, based upon induvidual answers of each document. 

This involves a lot of calls to the language model, but it does have the advantage that we can use an arbitrarily large number of documents.

Another limitation to it being slower due to many more calls to the LLM is that it can be worse if there is no answer because the question we ask has information spread across two documents and the knowledge is limited to one document at a time. It doesn't have  all information in the same context.

We can use LangChain plus to get access to LangSmith and see what's happening under the hood.

```python
# set environment variables
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "..."

# re-run the map-reduce chain

qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)
result = qa_chain_mr({"query": question})
result["result"]
```
In LangSmith UI we can see the run, and by selecting it we can see the input and the output. We can also see the child runs to analyze what's happening under the hood. Below, we see the the MapReduceChain involves four separate calls to the LLM. 

{{< figure src="./retrieval_qa_map_reduce_langsmith.png"  width="100%" >}}

If we click any of the calls, we will in-turn see that we have the input and the output for each of the documents.

Going back to our run, we can see that after it's run over each of these documents it's combined in a final chain, the StuffDocumentsChain where it stuffed all these responses into the final call. 

Clicking the StuffDocumentsChain we can see that we have the system message, the summaries of the four documents and the final answer.

{{< figure src="./retrieval_qa_map_reduce_langsmith_run.gif"  width="100%" >}}

#### **Refine**

We can do a similar thing using refine.

{{< figure src="./retrieval_qa_refine.png"  width="100%" >}}


```python
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="refine"
)
result = qa_chain_mr({"query": question})
result["result"]
```


{{< figure src="./retrieval_qa_refine_langsmith.png"  width="100%" >}}

We can see that it invokes the refineDocumentsChain which involves four sequential calls to an LLM chain.

Inside each call, we will see that there is system  message, user question and the answer. In the next call we will see that there is an additional user message behind the scenes saying " We have the opportunity to improve the existing answer if needed with some more context below". (Check langchain docs ```refineDocumentsChain``` for exact prompt). 

This refine chain performs better than the map_reduce chain as it enables flow of information, albeit sequentially putting more context for the LLM.


But what we created so far, the system doesn't have a concept of state. It doesn't remember what we had asked, the history. Which brings us to the next and final step of the process.

## Step 6 - Chat (with memory)

So far, we started with loading documents, splitting them, then we created embeddings and stored them in a vector store, we saw different ways of retrieval and shown that we can answer questions. For a fully functional chatbot, all that's left to do is to make sure we can have follow-up questions.

{{< figure src="./chat_history_retrieval.png"  width="100%" >}}

As we see above, we need to add a history component to maintain a conversation.

The interesting thing with langchain is that everything we've seen up until this point is modular and we can add anything we want to the workflow as seen below.

{{< figure src="./chat_history_modularity.png"  width="100%" >}}

One of the ways to add memory is to use ConversationBufferMemory from LangChain

```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

What this does is keep a list, a buffer of chat messages in history and it's going pass those along with the question to the chatbot every time. 

This can be used with a new type of chain, a ConversationalRetrievalChain. This chain adds a step to the retrieval QA chain process by taking the history and the new question and condensing it to a standalone question to be passed to the vector store to look up for documents.

```python
from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)
question = "Is probability a class topic?"
result = qa({"question": question})

result['answer']

question = "why are those prerequesites needed?"
result = qa({"question": question})

result['answer']

```
Using LangSmith, we can see the internal working of this chain. 

{{< figure src="./chat_history_langsmith.png"  width="100%" >}}

We can see an additional component, chat history being invoked along with the original question.

Let's see the end-to-end code, from the beginning now:


```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader


def load_db(file, chain_type, k):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa 


import panel as pn
import param

class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    
    def __init__(self,  **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        self.loaded_file = "docs/cs229_lectures/MachineLearning-Lecture01.pdf"
        self.qa = load_db(self.loaded_file,"stuff", 4)
    
    def call_load_db(self, count):
        if count == 0 or file_input.value is None:  # init or no file specified :
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
        else:
            file_input.save("temp.pdf")  # local copy
            self.loaded_file = file_input.filename
            button_load.button_style="outline"
            self.qa = load_db("temp.pdf", "stuff", 4)
            button_load.button_style="solid"
        self.clr_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def convchain(self, query):
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer'] 
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600, style={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''  #clears loading indicator when cleared
        return pn.WidgetBox(*self.panels,scroll=True)

    @param.depends('db_query ', )
    def get_lquest(self):
        if not self.db_query :
            return pn.Column(
                pn.Row(pn.pane.Markdown(f"Last question to DB:", styles={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("no DB accesses so far"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown(f"DB query:", styles={'background-color': '#F6F6F6'})),
            pn.pane.Str(self.db_query )
        )

    @param.depends('db_response', )
    def get_sources(self):
        if not self.db_response:
            return 
        rlist=[pn.Row(pn.pane.Markdown(f"Result of DB lookup:", styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    @param.depends('convchain', 'clr_history') 
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No History Yet")), width=600, scroll=True)
        rlist=[pn.Row(pn.pane.Markdown(f"Current Chat History variable", styles={'background-color': '#F6F6F6'}))]
        for exchange in self.chat_history:
            rlist.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    def clr_history(self,count=0):
        self.chat_history = []
        return 


# in-notebook UI 
cb = cbfs()

file_input = pn.widgets.FileInput(accept='.pdf')
button_load = pn.widgets.Button(name="Load DB", button_type='primary')
button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
button_clearhistory.on_click(cb.clr_history)
inp = pn.widgets.TextInput( placeholder='Enter text here…')

bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
conversation = pn.bind(cb.convchain, inp) 

tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation,  loading_indicator=True, height=300),
    pn.layout.Divider(),
)
tab2= pn.Column(
    pn.panel(cb.get_lquest),
    pn.layout.Divider(),
    pn.panel(cb.get_sources ),
)
tab3= pn.Column(
    pn.panel(cb.get_chats),
    pn.layout.Divider(),
)
tab4=pn.Column(
    pn.Row( file_input, button_load, bound_button_load),
    pn.Row( button_clearhistory, pn.pane.Markdown("Clears chat history. Can use to start a new topic" )),
    pn.layout.Divider(),
    pn.Row(jpg_pane.clone(width=400))
)
dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# ChatWithYourData_Bot')),
    pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3),('Configure', tab4))
)
dashboard
```