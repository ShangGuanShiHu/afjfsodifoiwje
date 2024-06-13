from llama_index.retrievers.bm25 import BM25Retriever as LlamaBM25Retriever

from llama_index.core.retrievers import BaseRetriever
from llama_index.core import (
    QueryBundle,
)
from llama_index.core.schema import NodeWithScore
from typing import List
import jieba


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def chinese_tokenizer(text: str) -> List[str]:
    tokens = jieba.lcut(text)
    chinese_stopwords = stopwords.words('chinese')
    return [token for token in tokens if token not in chinese_stopwords]



class BM25Retriever(BaseRetriever):
    def __init__(self, nodes, similarity_top_k: int = 3):
        self.bm25 = LlamaBM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=similarity_top_k,
            tokenizer=chinese_tokenizer
        )
        super().__init__()
        

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        node_with_scores: List[NodeWithScore] = await self.bm25.aretrieve(query_bundle)
        return node_with_scores

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        node_with_scores: List[NodeWithScore] = self.bm25.retrieve(query_bundle)
        return node_with_scores
