import asyncio
import os.path

import torch
from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.postprocessor import FlagEmbeddingReranker
from llama_index.llms.ollama import Ollama
from qdrant_client import models
from tqdm.asyncio import tqdm

from pipeline.retriever.bm25_retriever import BM25Retriever
from pipeline.retriever.fusion_retriever import FusionRetriever
from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers, save_pkl
from pipeline.retriever.qdrant_retriever import QdrantRetriever
from pipeline.rag import generation_with_knowledge_retrieval

retrieval_top_n = 30
rerank_top_n = 5

async def main():
    config = dotenv_values(".env")
    # 检查GPU是否可用并设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化 LLM 嵌入模型 和 Reranker
    llm = Ollama(
        model="qwen", base_url=config["OLLAMA_URL"], temperature=0, request_timeout=120
    )
    embeding = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-zh-v1.5",
        cache_folder="./",
        embed_batch_size=128,
        device=device
    )
    Settings.embed_model = embeding

    # 初始化 数据ingestion pipeline 和 vector store
    client, vector_store = await build_vector_store(config, reindex=False)

    collection_info = await client.get_collection(
        config["COLLECTION_NAME"] or "aiops24"
    )

    if collection_info.points_count == 0:
        data = read_data("data")
        pipeline = build_pipeline(llm, embeding, vector_store=vector_store)
        # 暂时停止实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        nodes = await pipeline.arun(documents=data, show_progress=True, num_workers=1)
        # 恢复实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )
        print(len(data))

    # 构建 BM25Retriever
    bm25_retriever = BM25Retriever(nodes, retrieval_top_n)
    # 构建 QdrantRetriever
    qdrant_retriever = QdrantRetriever(vector_store, embeding, similarity_top_k=retrieval_top_n)
    fusion_retriever = FusionRetriever([bm25_retriever, qdrant_retriever])
    queries = read_jsonl("question.jsonl")

    # 生成答案
    print("Start generating answers...")

    # ReRanker
    reranker = FlagEmbeddingReranker(
        top_n=rerank_top_n,
        model="BAAI/bge-reranker-large",
        use_fp16=False
    )
    results = []
    query_context_map = {}
    for query in tqdm(queries, total=len(queries)):
        context_str, result = await generation_with_knowledge_retrieval(
            query_str=query["query"],
            retriever=fusion_retriever,
            llm=llm,
            reranker=reranker
        )
        results.append(result)
        query_context_map[query["query"]] = context_str

    # 处理结果
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    save_answers(queries, results, "submit_result.jsonl")
    save_pkl("logs/context.pkl", query_context_map)


if __name__ == "__main__":
    asyncio.run(main())
