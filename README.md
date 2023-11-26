# nlp-assignment-wordweavers



## Pretraining Bert from scratch

This pretrained model is pushed to Hugging [Face](https://huggingface.co/Dhairya/nlp-bert-wordWeavers-pretrained)

Steps
1. The dataset is cleaned as it contains many empty strings and just heading.
2. To be able to train our model we need to convert our text into a tokenized format. Most Transformer models come with a pre-trained tokenizer, but since we are pre-training our model from scratch we also need to train a Tokenizer on our data. Transformer models often use subword tokenization, so a tokenizer needed training. The trained tokenizer is pushed on the 🤗 and can be found here Dhairya/nlp-bert-wordWeavers-tokenizer. The tokenized dataset has three element keys 'input_ids', 'token_type_ids', 'attention_mask’.
3. BERT usually accepts sequences of 512 tokens. We created a sequence of 255 tokens. We will concatenate the data such that we have a list of 255 tokens for each element in the tokenized dataset. ([CLS] + 255 (Sentence 1) + [SEP] 255 (Sentence 2)).
4. **Next Sentence Prediction (NSP):** NSP dataset is created such that 50% times sentence 2 comes after sentence 1. Rest 50% is random. To implement this logic we did when index is even the sentence 2 (which is the next sentence) comes after sentence 1 and for the odd index the two sentences are chosen at random. It added one extra feature in the dataset, ‘next_sentence_label’
5. **Masked Language Modelling (MLM):** The transformers library offers a streamlined solution for the MLM task. This is done using ‘DataCollatorForLanguageModeling’ with mlm probability 0.15 (similar in paper).
6. Pretraining was done for 5 epoch learning rate 0.01 using the Trainer class of hugging face. Here we loaded the bert architecture with random weights and this was pretrained on our dataset only. Below figure show the perplexity with each epoch

<img src="https://github.com/chirag-25/LLM-pertaining-finetuning/assets/76460264/2ec8e9ad-a36e-4335-87fe-370922157d42" alt="image" width="500">




