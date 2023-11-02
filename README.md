# CuriousGeorge-Query-Answerer
Query answerer module based on KEPLER and CQD for the CuriousGeorge LLM model with Knowledge Graph augmentation.

# About the Curious George Project
> Recent advancements in **Large Language Models (LLMs)** such as **OpenAI GPT, BERT, and LLaMA** have demonstrated their potential for complex reasoning tasks with natural language. However, there is still room for improvement as the inner workings of these models are not yet fully understood. In this line of thought, **Neurosymbolic Artificial Intelligence**, which combines **Symbolic Logic Reasoning** and **Deep Learning**, aims to create **explainable** inference models using the virtues of the two fields. **Knowledge Graphs (KGs)** are an essential component in this subject, since they provide concise representations of large knowledge bases, understandable for both users and models. Two significant challenges in this area are **query answering** from KGs, and the **integration of KG information** into the output of language models. 
> Researchers have proposed a variety of approaches to address these last two, including the use of Deep Learning for complex queries on KGs and **Augmented Language Models** that integrate recognition of entities from a KG. In this thesis, we propose to modify and combine these approaches with recent LLM developments, creating an explainable way for LLMs to work with **data from any KG**. Our approach will use LLMs for the KG entity embedding steps utilized in existing techniques, while keeping the other parts of the methods intact. Furthermore, we will use this new querying method for executing KG queries in the internal inference process of LLMs. Our goal is to reduce the frequency of hallucinations and attempt to make it easier to detect them by allowing the model to provide informed explanations, making them more broadly useful for general contexts.

# Complete Model Architecture
The complete Curious George model has an architecture based on that of the [ERNIE model](https://huggingface.co/docs/transformers/model_doc/ernie) with the addition of an extra *Query Answering module* for the model to be able to access Knowledge Graph data as needed during inference.
![ERNIE_basic_zoom_diag](https://github.com/Efrainq07/CuriousGeorge-Query-Answerer/assets/33973526/8ee15bb9-2c31-4985-9806-dce34972c035)

# Query Answerer Architecture
The specific part contained in this repo corresponds to the *Query Appendix*
![ERNIE_query](https://github.com/Efrainq07/CuriousGeorge-Query-Answerer/assets/33973526/670f78a1-f6fb-47e9-b227-4c4139598427)
