---
title: "LangChain"
date: 2024-2-6
lastmod: 2024-2-6
draft: true
garden_tags: ["large-language-models"]
summary: "Langchain ecosystem "
status: "seeding"
---

### LangChain - a framework for developing applications powered by language models.

LangChain provides standard interfaces and integrations for working with LLMs, retrieving application data, and directing agent behavior. It features composable components as well as pre-built chains for accomplishing common tasks. The goal is to simplify the process of leveraging LLMs to build context-aware, reasoning-based applications.
Some key capabilities provided by LangChain include:

• Interfacing with LLMs like GPT-3, Codex, and open source models
• Retrieving documents, databases, knowledge bases to provide context 
• Tools and templates to quickly build chatbots, QA systems, data analysis apps
• Monitoring, testing, and debugging chains
• Deploying chains via REST API or serverless functions

--------------------------

The components of Langchain are:
- **Prompts**
- **Chains**
- **Models**
- **Indexes**
- **Agents**

----------------------------

- For a large applications, prompts can be long and detailed. **Prompt templates** offer a useful abstraction to reuse good prompts when we can. 
- For complext applications using LLMs, we would sometimes want the output of the LLM to generate the ouput in a specific format. To achieve this, **output parsers** are provided.

--------------------------
## LCEL - LangChain Expression Language

Langchain composes chains of components. LCEL and the runnable protocol define:

- A set of allowed input types
- Requires methods like invoke, batch, stream,... 
- There are also ways of modifying the parameters at runtime like bind.... 
- Ouput types

We can do all this using the linux pipe syntax, which looks like this:

Chain = prompt | llm | OutputParser

More details about the interface that we expect all runnables to expose are detailed below.

Componets implement "Runnable" protocol.

Common methods include:

invoke - calls runnable on a single input
stream - calls runnable on a single input and streams back a response
batch - calls runnable on a list input

For all these synchronous methods, there is also an corresponding async method like ainvoke, astream, abatch

Common properties like:

- input_schema
- output_schema

Some common input and output types are:
| Component | Input Type | Output Type |
|----------|----------|----------|
| Prompt    | Dictionary     | Prompt Value     |
| Retriever    | Single String     | List of Documents     |
| LLM    | String, List of Messages or prompt value     | String     |
| ChatModel    | String, List of Messages or prompt value     | ChatMessage     |
| Tool    | String/Dictionary     | Tool Dependent      |
| Parser    | Output of LLM or ChatModel     | Parser Dependent     |


Advantages of using LCEL:

- Runnables Support:
    - Async, Batch and Streaming support
    - Fallbacks
    - Parallelism
        - LLM calls can be time consuming
        - Any components can be run in parallel
    - Logging is built in

----------------------------

An example of a chain with runnables, specifically a runnable map to supply user-provided inputs to the prompt.

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch

vectorstore = DocArrayInMemorySearch.from_texts(
    ["james worked at pendo", "bears like to eat honey"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()


retriever.get_relevant_documents("where did james work?")
retriever.get_relevant_documents("what do bears like to eat")

# NOW to use runnables

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""


prompt = ChatPromptTemplate.from_template(template)

from langchain.schema.runnable import RunnableMap

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser

chain.invoke({"question": "where did james work?"})

inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})

inputs.invoke({"question": "where did james work?"})
```

Another interesting use-case is for binding inputs to openAI functions

```python

functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    },
        {
      "name": "sports_search",
      "description": "Search for news of recent sport events",
      "parameters": {
        "type": "object",
        "properties": {
          "team_name": {
            "type": "string",
            "description": "The sports team to search for"
          },
        },
        "required": ["team_name"]
      }
    }
  ]

  model = model.bind(functions=functions)

  runnable = prompt | model
  runnable.invoke({"input": "how did the patriots do yesterday?"})
```


-------


One of the most features of LCEL is to implement fallbacks, not only on single inputs but entire sequences.

Let's create a situation which will fail using an older OpenAI model:

```python
from langchain.llms import OpenAI
import json

simple_model = OpenAI(
    temperature=0, 
    max_tokens=1000, 
    model="gpt-3.5-turbo-instruct"
)
simple_chain = simple_model | json.loads

challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"
simple_model.invoke(challenge)
# the next line will fail as the ouput is not actually valid json

simple_chain.invoke(challenge)
```

Now, to make a more newer version, that works and ouputs valid json:

```python
model = ChatOpenAI(temperature=0)
chain = model | StrOutputParser() | json.loads
chain.invoke(challenge)
final_chain = simple_chain.with_fallbacks([chain])
final_chain.invoke(challenge)
```

If we run above output,we can see that `chain` above outputs valid json. So, by creating a fallback method for the final chain using simple_chain and chain, we initially call the simple chain and if there is an error, the  chain (which is a part of the list in fallbacks) is called.