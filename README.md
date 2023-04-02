# BERTMODEL PREPROCESSING
Loading the dataset: The dataset is loaded from a online source.

Cleaning the data: This involves removing any irrelevant or noisy data, such as HTML tags, punctuation, and special characters.

Tokenization: This involves breaking down the text into individual tokens, usually words, which can be used as input to the model.

Text normalization: This step involves transforming the text to a standard format, such as lowercasing all letters or converting contractions to their expanded forms.

Stopword removal: This involves removing common words that do not add much meaning to the text, such as "the", "a", and "an".

Splitting into train, validation, and test sets: This involves dividing the dataset into separate subsets to train, validate, and test the model, respectively.

# ARCHITECTURE
I used a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model. Then fine-tuned the BERT model on our dataset using the Hugging Face transformers library in Python.

The BERT model consists of a stack of transformer encoder layers, which can be fine-tuned on downstream tasks by adding a task-specific output layer. In this case,I have added a single dense layer on top of the BERT model to classify the text into the appropriate category.

To fine-tune the BERT model, I trained the model on this dataset for several epochs, using binary cross-entropy loss as the objective function and the Adam optimizer to update the model weights.

# EVALUATION METRICS
I hvae used accuracy as the evaluation metric to measure the performance of my model. The accuracy is calculated by dividing the number of correctly classified samples by the total number of samples in the dataset.

After fine-tuning the BERT model on this dataset, I achieved an accuracy of 84% on the test set. This result indicates that my  model can accurately classify the text into the appropriate category.

# PERFORMANCE
The accuracy of 84% achieved by my  model is a good result, but there is always room for improvement. One possible way to improve the performance of the model is to fine-tune it on a larger dataset. Another way is to experiment with different pre-processing techniques or try using a different pre-trained language model.
