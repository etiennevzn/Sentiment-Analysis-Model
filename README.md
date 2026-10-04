# Sentiment Analysis Model

A simple sentiment analysis model implemented in **PyTorch**, designed to explore the fundamentals of text classification.

## Overview

The model predicts the sentiment of a sentence from its words. Each sentence is converted into a sequence of token IDs, mapped to learnable word embeddings, and then reduced to a single sentence representation by averaging the embeddings.

The resulting representation is passed through a linear layer and a `tanh` activation to produce a sentiment score between `-1` and `1`.

```text
Input sentence
      │
      ▼
Tokenization
      │
      ▼
Token IDs
      │
      ▼
Embedding Layer
      │
      ▼
Word Embeddings
      │
      ▼
Average Pooling
      │
      ▼
Linear Layer
      │
      ▼
Tanh
      │
      ▼
Sentiment Score [-1, 1]
```

## Model Architecture

The model is composed of three main components:

### 1. Embedding Layer

Each token is mapped to a learnable vector of fixed dimension.

```python
self.embedding_layer = nn.Embedding(vocabulary_size, embedding_dimension)
```

In the current implementation, the embedding dimension is set to `256`.

The embedding layer allows the model to learn a numerical representation of each word during training.

### 2. Average Pooling

The embeddings of all tokens in a sentence are averaged:

```python
averaged = torch.mean(embeddings, axis=1)
```

This produces a single fixed-size vector representing the entire sentence, regardless of its original length.

### 3. Linear Layer + Tanh

The averaged representation is projected to a single scalar:

```python
projected = self.linear_layer(averaged)
return self.tanh(projected)
```

The `tanh` activation maps the output to `(-1, 1)`, which matches the sentiment representation used by the model.

## Data Processing

The dataset is processed in several steps:

1. Extract all unique words from the training sentences.
2. Build a vocabulary mapping each word to an integer ID.
3. Reserve `0` for padding.
4. Convert each sentence into a sequence of token IDs.
5. Pad the sequences so that they have the same length.

For example:

```text
"best movie ever"
        ↓
[12, 47, 83]
        ↓
[12, 47, 83, 0, 0]
```

The padding allows sentences of different lengths to be processed together in batches.

## Training

The model is trained using the **Adam optimizer** and **Mean Squared Error (MSE)** loss.

At each training iteration, a random mini-batch of 64 examples is selected from the dataset:

```text
Full dataset
     │
     ▼
Random shuffle
     │
     ▼
64 examples
     │
     ▼
Forward pass
     │
     ▼
Compute MSE loss
     │
     ▼
Backpropagation
     │
     ▼
Adam parameter update
```

The current implementation performs 1000 training iterations.

## Example

After training, the model can be used to predict the sentiment of new sentences:

```python
examples = [
    "worst movie ever",
    "best movie ever",
    "weird but funny movie"
]

model.eval()

predictions = model(testing_tensor)

print(predictions.tolist())
```

The output is a continuous sentiment score:

```text
-1  ← strongly negative
 0  ← neutral
+1  ← strongly positive
```

## Current Limitations

Although this architecture is useful for understanding the basic principles of NLP models, it has several important limitations.

### No Word Order

Because all word embeddings are simply averaged, the model completely loses word order.

For example:

```text
"movie was not good"
"good was not movie"
```

would produce the same representation if they contain the same words.

More importantly, the model cannot properly capture relationships between words such as negation:

```text
"good"
"not good"
```

### No Contextual Word Representations

Each word has a single embedding regardless of its context.

For example, the word `"bank"` would have the same embedding in:

```text
"I went to the bank"
```

and:

```text
"The river bank was flooded"
```

A contextual model would represent these occurrences differently.

### Limited Handling of Long Sentences

Averaging all word embeddings gives every word roughly the same importance. Important words and irrelevant words contribute equally to the final representation.

### Simple Training Procedure

The current training loop randomly samples a single mini-batch at every iteration rather than explicitly iterating through the entire dataset over multiple epochs. A more standard training setup would use epochs and a `DataLoader` to iterate through all mini-batches.

## Possible Improvements

Several improvements could significantly increase the model's ability to understand language.

### 1. Attention Mechanism

Instead of simply averaging all word embeddings, an **attention mechanism** could learn which words are more important for predicting sentiment.

For example, in:

```text
"The movie was surprisingly good"
```

the model could assign greater importance to `"good"` and `"surprisingly"` than to `"the"`.

A simple attention mechanism could compute a weight for each token and use a weighted average:

```text
Word embeddings
      │
      ▼
Attention scores
      │
      ▼
Softmax weights
      │
      ▼
Weighted sum
      │
      ▼
Sentence representation
```

This would allow the model to focus on the most informative parts of a sentence.

### 2. Multi-Head Self-Attention

A more advanced approach would use **multi-head self-attention**, allowing each token to interact with the other tokens in the sentence.

This would help the model capture relationships such as:

```text
"The movie was not particularly good"
                  ↑
        "not" modifies "good"
```

Different attention heads can potentially learn different types of relationships between words.

### 3. Positional Encoding

Self-attention by itself does not inherently know the order of tokens.

**Positional encodings** could therefore be added to the token embeddings:

```text
Token embedding + positional encoding
                ↓
        Self-attention
```

This would allow the model to distinguish between different word orders and better understand sentence structure.

### 4. Transformer Architecture

The previous improvements naturally lead toward a small **Transformer encoder**.

A possible architecture would be:

```text
Token IDs
    ↓
Embedding
    +
Positional Encoding
    ↓
Transformer Encoder
    ↓
Sentence Representation
    ↓
Linear Layer
    ↓
Sentiment Score
```

This would provide a much richer representation than simple average pooling while remaining relatively lightweight.

### 5. Recurrent Architectures

Another possible direction would be to use an **RNN, GRU, or LSTM** instead of averaging the embeddings.

These architectures process the sequence step by step and can therefore retain information about the order of words.

### 6. Better Training Pipeline

The training procedure could also be improved by using PyTorch's `DataLoader`:

```python
DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)
```

This would provide a more standard training loop with explicit epochs and mini-batches.

Other improvements could include:

- validation and test sets
- early stopping
- learning-rate scheduling
- model checkpointing
- accuracy and other evaluation metrics
- hyperparameter tuning

## Future Work

The main direction for future development is to move from a **bag-of-embeddings approach** toward a model that explicitly captures relationships between tokens.

A natural progression would be:

```text
Average Embeddings
       ↓
Attention
       ↓
Multi-Head Self-Attention
       ↓
Positional Encoding
       ↓
Transformer Encoder
```

This progression would make it possible to explore how modern NLP architectures overcome the limitations of simple word averaging and progressively build more contextual representations of text.