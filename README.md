# Late Chunking Embedding

This is an implementation of a late chunking embedding, proposed in the paper [Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models](https://arxiv.org/abs/2409.04701) by [JinaAI](https://jina.ai/).

## Installation

```bash
git clone https://github.com/matheusjorge/late-chunking.git
cd late-chunking
pip install -e .
```

## Usage

```python
import torch
from transformers import AutoModel, AutoTokenizer

from late_chunking.embedding import LateChunkingEmbedding

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=8192)
model = AutoModel.from_pretrained(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
    safe_serialization=True,
    rotary_scaling_factor=2,
)
_ = model.eval()
sep_token = "<sep>"
test_sentences = [
    "This is an example sentence",
    "This is another example sentence",
    "Lets try a huge sentence that will be split into chunks",
]


embedder = LateChunkingEmbedding(model, tokenizer, sep_token)

embeddings = embedder.encode(test_sentences)
```