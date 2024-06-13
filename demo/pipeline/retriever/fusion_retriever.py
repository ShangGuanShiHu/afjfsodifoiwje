from typing import List
from llama_index.core import (
    QueryBundle,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore



class FusionRetriever(BaseRetriever):
    def __init__(self, 
                 retrievers: List[BaseRetriever]):
        self._retrievers = retrievers
        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        combined_results = []
        for retriever in self._retrievers:
            res = await retriever._aretrieve(query_bundle)
            combined_results.extend(res)

        return combined_results

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        combined_results = []
        for retriever in self._retrievers:
            combined_results += retriever._retrieve(query_bundle)

        return combined_results
