# Transformers

A basic transformer package. This repository is a copy of one by [Lior Sinai](https://github.com/LiorSinai/TransformersLite.jl)
and is paired with this [blog post](https://liorsinai.github.io/coding/2022/05/18/transformers.html). For a much more comprehensive package with APIs for HuggingFace, optimizations and more, please see Transformers.jl at [github.com/chengchingwen/Transformers.jl](https://github.com/chengchingwen/Transformers.jl).

This package is designed to work with [Flux](https://github.com/FluxML/Flux.jl). It provides a multi-head attention layer as described in the paper [Attention is all you need](https://arxiv.org/abs/1706.03762).
It also provides 
- A simple index tokenizer for mapping words to indices.
- A wrapper for an embedding layer.
- A wrapper for a mean layer.
- A position encoding layer.
- Two encompassing layers to chain these together: `TransformerEncoderBlock` and `TransformerClassifier`. Flux's `chain` function can also be used to chain the layers together.

Two implementations are provided for the 4D batch multiplication such that `A×B` results in `C[:,:,k,l] == A[:,:,k,l] * B[:,:,k,l]`.
These are `mul4d` and an extension to NNlib's `batched_mul`. The extension to `batched_mul` is about 1.5× faster than `mul4d`.

An example model output looks like:
```
TransformerClassifier(
  Embed((32, 7455)),                    # 238_560 parameters
  PositionEncoding(32),
  Dropout(0.1),
  TransformerEncoderBlock(
    MultiheadAttention(num_heads=4, head_size=8, 32=>32)(
      denseQ = Dense(32 => 32),         # 1_056 parameters
      denseK = Dense(32 => 32),         # 1_056 parameters
      denseV = Dense(32 => 32),         # 1_056 parameters
      denseO = Dense(32 => 32),         # 1_056 parameters
    ),
    Dropout(0.1),
    LayerNorm(32),                      # 64 parameters
    Dense(32 => 128, relu),             # 4_224 parameters
    Dense(128 => 32),                   # 4_128 parameters
    Dropout(0.1),
    LayerNorm(32),                      # 64 parameters
  ),
  Dense(32 => 1),                       # 33 parameters
  FlattenLayer(),
  Dense(50 => 5),                       # 255 parameters
)        # Total: 21 trainable arrays, 251_552 parameters,
          # plus 1 non-trainable, 32_000 parameters, summarysize 1.083 MiB
```
Please see the [example](/examples/) folder for utility functions, notebooks and a training script which demonstrate the module's capabilities.
These examples use tokenizers from my TokenizersLite repository at [https://github.com/LiorSinai/TokenizersLite](https://github.com/LiorSinai/TokenizersLite).
However any compatible tokenizer can be used.

To run the examples:
```bash
mkdir outputs
python examples/download_amazon_reviews.py
julia examples/demo.jl --threads auto
### after training completes
jupyter notebook
```

## Case study

The use case of Amazon Reviews from [Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) was investigated.
The task was to predict the star rating on a 5 star scale given a review. 

It should be noted that this task can be solved with simpler models. A TFIDF model paired with logistic regression with approximately 10,000 weights
achieved similar accuracy to these models with more than 240,000 weights.

The accuracy achieved was around 50% for the 5 star classification task.

### 5 star classification task
<img src="images/confusion_matrix_classification5.png"
     alt="confusion matrix"
    />

Looking at the confusion matrix for the 5 star classification, we can see that the model struggles more with the middle ratings of 2-4.
Again this is hypothesized  to be partially because of inconsistencies in the underlying data.

<img src="images/predictions_classification5.png"
     alt="bar chart predication vs ground truth"
    />

Seeing in another view as a bar chart, for each star the most likely prediction is the star itself.
However the distributions do have a spread and have significant overlaps of confusion.

