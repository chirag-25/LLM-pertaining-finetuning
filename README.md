## Pre-Training Bert from scratch

This pretrained model is pushed to Hugging [Face](https://huggingface.co/Dhairya/nlp-bert-wordWeavers-pretrained)

Steps
1. The dataset is cleaned as it contains many empty strings and just heading.
2. To be able to train our model we need to convert our text into a tokenized format. Most Transformer models come with a pre-trained tokenizer, but since we are pre-training our model from scratch we also need to train a Tokenizer on our data. Transformer models often use subword tokenization, so a tokenizer needed training. The trained tokenizer is pushed on the 🤗 and can be found here Dhairya/nlp-bert-wordWeavers-tokenizer. The tokenized dataset has three element keys 'input_ids', 'token_type_ids', 'attention_mask’.
3. BERT usually accepts sequences of 512 tokens. We created a sequence of 255 tokens. We will concatenate the data such that we have a list of 255 tokens for each element in the tokenized dataset. ([CLS] + 255 (Sentence 1) + [SEP] 255 (Sentence 2)).
4. **Next Sentence Prediction (NSP):** NSP dataset is created such that 50% times sentence 2 comes after sentence 1. Rest 50% is random. To implement this logic we did when index is even the sentence 2 (which is the next sentence) comes after sentence 1 and for the odd index the two sentences are chosen at random. It added one extra feature in the dataset, ‘next_sentence_label’
5. **Masked Language Modelling (MLM):** The transformers library offers a streamlined solution for the MLM task. This is done using ‘DataCollatorForLanguageModeling’ with mlm probability 0.15 (similar in paper).
6. Pretraining was done for 5 epoch learning rate 0.01 using the Trainer class of hugging face. Here we loaded the bert architecture with random weights and this was pretrained on our dataset only. Below figure show the perplexity with each epoch

<img src="https://github.com/chirag-25/LLM-pertaining-finetuning/assets/76460264/2ec8e9ad-a36e-4335-87fe-370922157d42" alt="image" width="500">

## Fine-Tunning our pretrained model for Classification on the SST-2 Dataset for sentiment analysis.

Steps: 
1. We created an 80:20 split of the SST-2 Dataset. To ensure a balanced dataset, we performed stratified sampling with random seed=1. The fine-tuning was performed on the train split.
2. The pretrained model and the tokenizer are firsted loaded into the system using the transformers library and the AutoModelForSequenceClassification module and the number of classes was set to 2.
3. We then tokenize the dataset using a helper function and the loaded pre-trainer tokenizer.
4. We then configured the training arguments which will be used to initialize the trainer. We appropriately choose the hyperparameters to fine-tune the model in the most efficient way possible.
5. A compute_metrics function was defined, which again will be fed into the trainer and will be used to compute the precision, recall, f1 and accuracy scores of the validation set at the end of each epoch during training.
6. We then created a trainer instance, feeding the tokenized train and test dataset into the instance.
7. We then ran the trainer and set up a function in order to print the final returns of the fine-tuned model on the test dataset. It is pushed to [Hugging Face](https://huggingface.co/Dhairya/bert-wordweavers-ft-sst2)

### Evaluation
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.5092 |
| Precision | 0.2593 |
| Recall    | 0.5092 |
| F1 Score  | 0.3436 |

These results are obtained by evaluating our fine tuned model on the fully unseen Validation Split of the SST2 dataset. The Test split was corrupted with all labels marked as -1 in a 0 vs 1 classification problem. So, we went ahead with evaluation on the Validation set. The above metrics are computed by taking the average of class-wise scores for these metrics. The weighted average is on the basis of distribution of each class in the evaluation dataset. 


## Fine-Tunning our pretrained model for Question-Answering on the SQuAD dataset.
1. We created an 80:20 split of the SQuAD dataset again  by using stratified sampling on the titles and combining the train and validation dataset. The fine-tuning was performed on the train split.
2. The following steps were used to fine-tune the pre-trained model on SQUAD for context based question answering:
3. We first tokenize the data by passing it through our loaded pre-trained tokenizer.
4. Since the dataset does not contain the end token indices of the answers, we calculate and add them to the answers list using the start index and the answer sentence.
5. We correct any off-set errors that might be there in the dataset and remove any data points which have missing questions or answers.
6. We define a dataset trainer argument(hyperparameter) including the number of epochs, learning rates etc and feed it into the trainer.
7. Lastly we run a preliminary evaluation on the validation data set using the trainer instance. It is pushed to [Hugging Face](https://huggingface.co/Dhairya/bert-wordweavers-ft-squad)

### Evaluation

| Metric      | Score  |
|-------------|--------|
| F1          | 0.0024 |
| BLEU        | 0.0174 |
| METEOR      | 0.0244 |
| Exact Match | 0.0013 |





