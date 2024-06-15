from typing import List
import qdrant_client

from llama_index.core.llms.llm import LLM
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import (
    QueryBundle,
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.base.llms.types import CompletionResponse
from custom.template import QA_TEMPLATE, HYDE_TEMPLATE
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine


# async def hyde(
#     query_str: str,
#     retriever: BaseRetriever,
#     llm: LLM,
#     qa_template: str = QA_TEMPLATE,
#     reranker: BaseNodePostprocessor | None = None,
# ):
#     '''
#         HYDE
#         https://mp.weixin.qq.com/s/I-PEO_mgH5OO-qzvLSQC1A
#     '''
#     hyde_prompt = PromptTemplate(HYDE_TEMPLATE).format(
#         query_str=query_str
#     )
#     try:
#         ret = await llm.acomplete(hyde_prompt)
#     except:
#         ret = CompletionResponse(text=query_str)
#     answer_wo_doc = ret.text
#     answer_bundle = QueryBundle(query_str=answer_wo_doc)
#     node_with_scores = await retriever.aretrieve(answer_bundle)
#     if reranker:
#         node_with_scores = reranker.postprocess_nodes(node_with_scores, answer_bundle)
#     context_str = "\n\n".join(
#         [f"{node.metadata['document_title']}: {node.text}" for node in node_with_scores]
#     )
#     fmt_qa_prompt = PromptTemplate(qa_template).format(
#         context_str=context_str, query_str=query_str
#     )
#     try:
#         ret = await llm.acomplete(fmt_qa_prompt)
#     except:
#         ret = CompletionResponse(
#                     text="不确定"
#                 )
#     return context_str, ret


async def generation_with_knowledge_retrieval(
        query_str: str,
        retriever: BaseRetriever,
        llm: LLM,
        qa_template: str = QA_TEMPLATE,
        reranker: BaseNodePostprocessor | None = None,
        debug: bool = False,
        progress=None,
) -> CompletionResponse:
    query_boundle = QueryBundle(query_str=query_str)

    node_with_scores = await retriever.aretrieve(query_boundle)
    if debug:
        print(f"retrieved:\n{node_with_scores}\n------")

    '''
        ReRank
        https://blog.csdn.net/littleblack201608/article/details/136518983
    '''
    if reranker:
        node_with_scores = reranker.postprocess_nodes(node_with_scores, query_boundle)
        if debug:
            print(f"reranked:\n{node_with_scores}\n------")
    context_str = "\n\n".join(
        [f"{node.metadata['document_title']}: {node.text}" for node in node_with_scores]
    )
    fmt_qa_prompt = PromptTemplate(qa_template).format(
        context_str=context_str, query_str=query_str
    )
    try:
        ret = await llm.acomplete(fmt_qa_prompt)
        if ret.text == "不确定" or "无法确定" in ret.text:
            return hyde_generation(query_str, llm, retriever, qa_template, reranker)
    except:
        return hyde_generation(query_str, llm, retriever, qa_template, reranker)

    if progress:
        progress.update(1)
    return context_str, ret


async def hyde_generation(
        query_str: str,
        llm: LLM,
        retriever: BaseRetriever,
        qa_template: str = QA_TEMPLATE,
        reranker: BaseNodePostprocessor | None = None,
        progress=None
):
    my_hyde_prompt = PromptTemplate(
        HYDE_TEMPLATE
    )
    hyde = HyDEQueryTransform(llm=llm,
                              hyde_prompt=my_hyde_prompt,
                              include_original=True)
    query_boundle = hyde(QueryBundle(query_str))
    node_with_scores = await retriever.aretrieve(query_boundle)
    if reranker:
        node_with_scores = reranker.postprocess_nodes(node_with_scores, query_boundle)
    context_str = "\n\n".join(
        [f"{node.metadata['document_title']}: {node.text}" for node in node_with_scores]
    )
    fmt_qa_prompt = PromptTemplate(qa_template).format(
        context_str=context_str, query_str=query_str
    )
    try:
        ret = await llm.acomplete(fmt_qa_prompt)
    except:
        ret = CompletionResponse(text='不确定')

    if progress:
        progress.update(1)
    return context_str, ret



