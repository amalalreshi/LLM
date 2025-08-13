# LLM
Langchain and RAG Technique Project 

# 📚 PDF Question Answering with LangChain & ALLaM

## 🚀 Overview
This project demonstrates how to build a **PDF Question Answering** system using [LangChain](https://www.langchain.com/), semantic search, and a **fully open-source large language model (LLM)**.

The basic idea:
1. Read a PDF document.
2. Split the text into smaller chunks.
3. Convert the text into embeddings (vector representations).
4. Use semantic search to find the most relevant chunks.
5. Generate answers in **Arabic** or **English**.

---

## 🛠 Initial Approach (OpenAI API)
The first version used **OpenAI** for both embeddings and the LLM:

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI

##Problem
Using OpenAI requires:

A valid API key (OPENAI_API_KEY).

Internet access to call their servers.

Pay-per-request charges.

If the API key is missing, expired, or incorrect, the app will fail.
Additionally, OpenAI models are closed-source, meaning no local hosting or customization.
