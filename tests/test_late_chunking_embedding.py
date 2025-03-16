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


class TestLateChunkingEmbedding:
    def test_encode_default(self):
        embeddings = embedder.encode(test_sentences)
        assert embeddings.shape == (3, 768)

    def test_encode_with_prefix(self):
        embeddings = embedder.encode(test_sentences, prefix="query_document")
        assert embeddings.shape == (3, 768)


class TestLateChukingSentenceGeneration:
    def test_one_sentence(self):
        sentences, sentences_indexes = embedder._create_model_inputs(
            test_sentences,
            max_tokens=100,
        )
        assert len(sentences) == 1
        assert len(sentences_indexes) == 1

    def test_two_sentences(self):
        sentences, sentences_indexes = embedder._create_model_inputs(
            test_sentences,
            max_tokens=15,
        )
        assert len(sentences) == 2
        assert len(sentences_indexes) == 2

    def test_three_sentences(self):
        sentences, sentences_indexes = embedder._create_model_inputs(
            test_sentences,
            max_tokens=1,
        )
        assert len(sentences) == 3
        assert len(sentences_indexes) == 3


class TestMatryoshkaEmbedding:
    def test_full_embedding(self):
        embeddings = embedder.encode(test_sentences)
        assert embeddings.shape == (3, 768)

    def test_partial_512(self):
        embeddings = embedder.encode(test_sentences, dim=512)
        assert embeddings.shape == (3, 512)

    def test_partial_256(self):
        embeddings = embedder.encode(test_sentences, dim=256)
        assert embeddings.shape == (3, 256)


class TestNormalization:
    def test_without_normalization(self):
        embeddings = embedder.encode(test_sentences, normalize=False)
        assert embeddings.shape == (3, 768)
        assert not torch.allclose(embeddings.norm(dim=1), torch.ones(3))

    def test_with_normalization(self):
        embeddings = embedder.encode(test_sentences, normalize=True)
        assert embeddings.shape == (3, 768)
        assert torch.allclose(embeddings.norm(dim=1), torch.ones(3))

    def test_partial_embedding_without_normalization(self):
        embeddings = embedder.encode(test_sentences, dim=512, normalize=False)
        assert embeddings.shape == (3, 512)
        assert not torch.allclose(embeddings.norm(dim=1), torch.ones(3))

    def test_partial_embedding_with_normalization(self):
        embeddings = embedder.encode(test_sentences, dim=512, normalize=True)
        assert embeddings.shape == (3, 512)
        assert torch.allclose(embeddings.norm(dim=1), torch.ones(3))
