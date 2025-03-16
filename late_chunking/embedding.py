from copy import copy
from typing import Dict, List, Tuple

import torch
from torch.nn import functional as F

from .protocols import Model, Tokenizer


class LateChunkingEmbedding:
    def __init__(self, model: Model, tokenizer: Tokenizer, separator: str):
        self.model = model
        self.tokenizer = tokenizer
        self.separator = separator
        encoded_sep = self.tokenizer(
            separator, padding=True, truncation=True, return_tensors="pt"
        )
        # Remove the first and last token from sequence
        self.separator_tokens = encoded_sep["input_ids"][0][1:-1]

    def encode(
        self,
        texts: List[str],
        max_tokens: int = 2048,
        normalize: bool = True,
        prefix: str | None = None,
        dim: int | None = None,
    ) -> torch.Tensor:
        sentences, sentences_indexes = self._create_model_inputs(
            texts, max_tokens, prefix=prefix
        )
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sep_indexes = self._find_sep_indexes(encoded_input, self.separator_tokens)

        sentence_embeddings = self._mean_pooling(
            model_output, encoded_input["attention_mask"], sep_indexes
        )

        flattened_sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        flattened_sentence_indexes = [
            i for sentence_indexes in sentences_indexes for i in sentence_indexes
        ]

        # If the same text is present in multiple sentences, we average the embedding
        grouped_embeddings = {}
        for index, embedding in zip(
            flattened_sentence_indexes, flattened_sentence_embeddings
        ):
            if index not in grouped_embeddings:
                grouped_embeddings[index] = [embedding]
            else:
                grouped_embeddings[index].append(embedding)

        for index, embeddings in grouped_embeddings.items():
            grouped_embeddings[index] = torch.stack(embeddings).mean(dim=0)

        # Get the embedding ordered by dict keys, just to be sure
        sorted_embeddings = sorted(grouped_embeddings.items(), key=lambda x: x[0])
        embeddings = torch.stack([embedding for _, embedding in sorted_embeddings])

        if dim:
            embeddings = embeddings[:, :dim]

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def _count_tokens_per_text(self, texts: List[str]) -> torch.Tensor:
        # Remove the first and last token for counting tokens
        return (
            self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")[
                "attention_mask"
            ].sum(dim=1)
            - 2
        )

    def _create_model_inputs(
        self,
        texts: List[str],
        max_tokens: int = 2048,
        context_overlap: float = 0.1,
        prefix: str | None = None,
    ) -> Tuple[List[str], List[List[int]]]:
        sep_length = len(self.separator_tokens)
        if prefix:
            prefix_token_count = self.tokenizer(
                [prefix], padding=True, truncation=True, return_tensors="pt"
            )["attention_mask"][0].sum()
            max_tokens -= prefix_token_count
        text_tokens_counts = self._count_tokens_per_text(texts)

        # Initialize state variables
        current_index = 1
        current_token_count = text_tokens_counts[0]
        current_sentence_indexes = [0]
        current_sentence = texts[0]

        # Objects to return
        sentences_indexes: List[List[int]] = []
        sentences: List[str] = []

        while current_index < len(texts):
            current_count = text_tokens_counts[current_index]

            # Check if the current sentence can be added to the context
            # We subtract 2 because the first and last tokens are not included in the count
            if current_token_count + current_count + sep_length <= max_tokens - 2:
                current_sentence_indexes.append(current_index)
                current_sentence += f" {self.separator} {texts[current_index]}"
                current_token_count += current_count + sep_length
            else:
                sentences_indexes.append(copy(current_sentence_indexes))
                sentences.append(copy(current_sentence))

                current_sentence_indexes = [current_index]
                current_sentence = texts[current_index]
                current_token_count = text_tokens_counts[current_index]
            current_index += 1

        sentences.append(copy(current_sentence))
        sentences_indexes.append(copy(current_sentence_indexes))

        if prefix:
            sentences = [f"{prefix} {sentence}" for sentence in sentences]
        return sentences, sentences_indexes

    def _find_sep_indexes(
        self, encoded_input: Dict[str, torch.Tensor], sep_sequence: List[int]
    ) -> List[List[Dict[str, int]]]:
        texts_splits = []
        for tokens in encoded_input["input_ids"]:
            splits = []
            current_split = {"start": 0, "end": 0}
            len_tokens = len(tokens)

            for i in range(len_tokens - 2):
                match = True
                for j, token in enumerate(sep_sequence):
                    if tokens[i + j] != token:
                        match = False
                        break
                if not match:
                    continue
                current_split["end"] = i - 1
                splits.append(copy(current_split))
                current_split["start"] = i + len(sep_sequence)

            current_split["end"] = len_tokens - 2
            splits.append(copy(current_split))
            texts_splits.append(splits)
        return texts_splits

    def _mean_pooling(
        self,
        model_output: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        sep_indexes: List[List[Dict[str, int]]],
    ) -> List[torch.Tensor]:
        texts_embeddings = []
        for i, token_embeddings in enumerate(model_output[0]):
            input_mask_expanded = (
                attention_mask[i].unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            chunks = []
            for split in sep_indexes[i]:
                split_embedding = token_embeddings[split["start"] : split["end"] + 1]
                split_mask = input_mask_expanded[split["start"] : split["end"] + 1]
                chunk_embedding = torch.sum(
                    split_embedding * split_mask, 0
                ) / torch.clamp(split_mask.sum(0), min=1e-9)
                chunks.append(chunk_embedding)
            texts_embeddings.append(torch.stack(chunks))
        return texts_embeddings
