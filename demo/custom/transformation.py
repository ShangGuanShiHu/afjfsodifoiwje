from typing import Optional, Sequence
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.schema import BaseNode
from llama_index.llms.ollama import Ollama
from llama_index_client import TextNode
from llama_index.core import PromptTemplate
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from typing import Any, Dict, List
from llama_index.legacy.llms import OpenAILike as OpenAI
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms.llm import LLM
from llama_index.core.service_context_elements.llm_predictor import (
    LLMPredictorType,
)
from snownlp import SnowNLP


class CustomFilePathExtractor(BaseExtractor):
    last_path_length: int = 4

    def __init__(self, last_path_length: int = 4, **kwargs):
        super().__init__(last_path_length=last_path_length, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "CustomFilePathExtractor"

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        metadata_list = []
        for node in nodes:
            node.metadata["file_path"] = "/".join(
                node.metadata["file_path"].split("/")[-self.last_path_length:]
            )
            metadata_list.append(node.metadata)
        return metadata_list


class CustomTitleExtractor(BaseExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "CustomTitleExtractor"

    # 将Document的第一行作为标题
    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:

        try:
            document_title = nodes[0].text.split("\n")[0]
            last_file_path = nodes[0].metadata["file_path"]
        except:
            document_title = ""
            last_file_path = ""
        metadata_list = []
        for node in nodes:
            if node.metadata["file_path"] != last_file_path:
                document_title = node.text.split("\n")[0]
                last_file_path = node.metadata["file_path"]
            node.metadata["document_title"] = document_title
            metadata_list.append(node.metadata)

        return metadata_list


DEFAULT_SUMMARY_EXTRACT_TEMPLATE = """\
Here is the content of the section:
{context_str}

Summarize the key topics and entities of the section. \

Summary: """


class CustomSummaryExtractor(BaseExtractor):
    """
    Summary extractor. Node-level extractor with adjacent sharing.
    Extracts `section_summary`, `prev_section_summary`, `next_section_summary`
    metadata fields.

    Args:
        llm (Optional[LLM]): LLM
        summaries (List[str]): list of summaries to extract: 'self', 'prev', 'next'
        prompt_template (str): template for summary extraction
    """

    llm: Ollama = Field(description="The LLM to use for generation.")
    summaries: List[str] = Field(
        description="List of summaries to extract: 'self', 'prev', 'next'"
    )
    prompt_template: str = Field(
        description="Template to use when generating summaries.",
    )

    _self_summary: bool = PrivateAttr()
    _prev_summary: bool = PrivateAttr()
    _next_summary: bool = PrivateAttr()

    def __init__(
            self,
            llm: Optional[LLM] = None,
            # TODO: llm_predictor arg is deprecated
            llm_predictor: Optional[LLMPredictorType] = None,
            summaries: List[str] = ["self"],
            prompt_template: str = DEFAULT_SUMMARY_EXTRACT_TEMPLATE,
            num_workers: int = DEFAULT_NUM_WORKERS,
            **kwargs: Any,
    ):
        # validation
        if not all(s in ["self", "prev", "next"] for s in summaries):
            raise ValueError("summaries must be one of ['self', 'prev', 'next']")
        self._self_summary = "self" in summaries
        self._prev_summary = "prev" in summaries
        self._next_summary = "next" in summaries

        super().__init__(
            llm=llm,
            summaries=summaries,
            prompt_template=prompt_template,
            num_workers=num_workers,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SummaryExtractor"

    async def _agenerate_node_summary(self, node: BaseNode) -> str:
        # """Generate a summary for a node."""
        # if self.is_text_node_only and not isinstance(node, TextNode):
        #     return ""

        context_str = node.get_content(metadata_mode=self.metadata_mode)

        # 用大模型提取摘要
        # summary = await self.llm.apredict(
        #     PromptTemplate(template=self.prompt_template), context_str=context_str
        # )

        # 用TextRank算法提取摘要
        s = SnowNLP(context_str)
        summary = s.summary(1)[0]

        return summary.strip()

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        # if not all(isinstance(node, TextNode) for node in nodes):
        #     raise ValueError("Only `TextNode` is allowed for `Summary` extractor")

        node_summaries_jobs = []
        for node in nodes:
            node_summaries_jobs.append(self._agenerate_node_summary(node))

        node_summaries = await run_jobs(
            node_summaries_jobs,
            show_progress=self.show_progress,
            workers=1,
        )

        # Extract node-level summary metadata
        metadata_list: List[Dict] = [{} for _ in nodes]
        for i, metadata in enumerate(metadata_list):
            if i > 0 and self._prev_summary and node_summaries[i - 1]:
                metadata["prev_section_summary"] = node_summaries[i - 1]
            if i < len(nodes) - 1 and self._next_summary and node_summaries[i + 1]:
                metadata["next_section_summary"] = node_summaries[i + 1]
            if self._self_summary and node_summaries[i]:
                metadata["section_summary"] = node_summaries[i]

        return metadata_list

