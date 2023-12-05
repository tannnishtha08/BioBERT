# Tagging Genes and Proteins with BioBERT

Text mining in the clinical domain has become increasingly important with the number of biomedical documents currently out there with valuable information waiting to be deciphered and optimized by NLP techniques. With the accelerated progress in NLP, pre-trained language models now carry millions (or even billions) of parameters and can leverage massive amounts of textual knowledge for downstream tasks such as question answering, natural language inference, and in the case that we will work through, biomedical text tagging via named-entity recognition.

## TASK
Named-entity recognition (NER) is the recognition process of numerous proper nouns (or given target phrases) that we establish as entity types to be labeled. The datasets used to evaluate NER are structured in the BIO (Beginning, Inside, Outside) schema, the most common tagging format for sentence tokens within this task. Additionally, an “S” like in “S-Protein” can be used to infer a single token. That way we can note the positional prefix and entity type that is being predicted from the training data.

![image](https://github.com/tannnishtha08/BioBERT/assets/98115816/d5c85f5e-461c-4d4f-8d76-df7d82cbd72c)
