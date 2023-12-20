---
title: "HuggingFace NLP"
date: 2023-12-05
lastmod: 2023-12-05
draft: false
garden_tags: ["natural_language_processing"]
summary: " "
status: "seeding"
links:
    external_link:
        text: "HuggingFace NLP course"
        icon: "fas fa-external-link-alt"
        href: "https://huggingface.co/learn/nlp-course/chapter0/1?fw=pt"
        weight: 1    
---

# These are some notes on understanding the HuggingFace NLP ecosystem.

Natural language processing is the part of AI dealing with everything related to human language. In recent times, NLP has began to rise as one of the fastest growing sub-fields within AI, accelerated by innovations that have happened within the past 10 years. NLP is also part of multi-modal challenges, such as generating a transcript of an audio sample or a description of an image.



| A list of some common NLP tasks include:                      |
|------------------------------------|---------------------------------|
| Sentiment analysis                 | Machine translation             |
| Automatic summarization            | Document classification         |
| Lexical analysis                   | Natural language generation      |
| Named-entity recognition           | Speech recognition              |
| Part-of-speech tagging             | Information retrieval           |


The inherent challenge of NLP is context. Consider these sentences:

**Sentence 1**: "She planted a tree near the bank."

**Sentence 2**: "She withdrew money from the bank."

In these two sentences, the word "bank" has entirely different meanings based on the context:

In the first sentence, "bank" likely refers to the side of a river or a financial institution where you deposit money. The context suggests an action related to nature or landscaping.

In the second sentence, "bank" refers to a financial institution where you withdraw money. The context here is related to banking and finance.


Transformers are part of the latest algorithms that have been able to tackle this problem more efficiently than ever before, and HuggingFace began with the aim of easing access and use of transformer-based models.

The transformers library which can be accessed through a simple one-liner:

```python
import transformers
```
The most basic object in the 🤗 Transformers library is the pipeline() function. It connects a model with its necessary preprocessing and postprocessing steps, allowing us to directly input any text and get an intelligible answer:

```python 
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("A briefer HuggingFace course? This could be a useful reference!")
```

[Transformer architecture](https://arxiv.org/abs/1706.03762) was introduced in June 2017, deriving from the attention-mechanism (which deserves its own post) used for machine translation. Since then, there have been thousands of models released based upon the architecture and adaptions with more known examples being the GPT family, BERT, RoBERTa, T5, Claude and more.