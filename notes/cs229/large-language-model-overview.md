# Building Large Language Models

## What matters in training LLMs:
1. Architecture
2. Training Algorithms / Loss
3. Data
4. Evaluation
5. System / Hardware

Architecture and Training Algorithms is what academia focuses on the most. However, in practive, Data, Evaluation and System the Model will run on matters more. So industry is more focused on the later 3 where as academia is more focused on the former 2.

## Pretraining and Posttraining:
### Pretraining:
Pretraining is the process of training a language model on a large corpus of text to learn general language patterns and representations. This allows the model to capture the statistical properties of natural language and improve its performance on downstream tasks.
Example: GPT-3, BERT, RoBERTa, etc.

### Posttraining:
Posttraining, also known as fine-tuning, is the process of adapting a pre-trained language model to a specific task or domain. This involves training the model on a smaller dataset specific to the task, which allows it to learn task-specific patterns and improve its performance on that task.
Example: ChatGPT, Claude, etc.

## Language Model:
- Probability Distribution over a sequence of tokens/words p(x_1, x_2, ..., x_L)
- LMs are generative models: x_1:L ~ p(x_1, x_2, ..., x_L)

### AutoRegressive Models:
- Predicts the next token in a sequence based on the previous tokens

p(x_1, ..., x_L) = p(x_1).p(x_2|x_1).p(x_3|x_1, x_2). ... .p(x_L|x_1, ..., x_{L-1})

- No approximation, just chain rule of probability. This means that your model needs to predict only the next token based on the past context. This is a key property of autoregressive models, as they can generate text by predicting one token at a time, conditioned on the previous tokens. This allows them to generate text that is coherent and contextually relevant.

- Task: Predict the next word in a sentence
1. Tokenize
2. forward
3. predict the probability of next token
4. sample
5. detokenize

### Tokenizer:
- more general than words (eg. caligraphic languages, typos etc.)
- shorter sequences than with characters
- Example: Byte Pair Encoding ***(BPE)***

### How to train Tokenizer:
1. Take large corpus of text
2. Start with one token per character
3. Merge common pairs of tokens into a token
4. Repeat until desired vocab size or all merged

LLM Evaluation: Perplexity
- Idea: Validation Loss
