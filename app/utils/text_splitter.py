"""文本切片模块 - 使用 LangChain"""
from typing import List
from loguru import logger
from app.config import settings
from langchain.text_splitter import RecursiveCharacterTextSplitter as LangChainRecursiveCharacterTextSplitter


class RecursiveCharacterTextSplitter:
    """递归字符文本分割器 - 基于 LangChain"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", " ", ""]
        
        # 使用 LangChain 的文本分割器
        self._splitter = LangChainRecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )

    def split_text(self, text: str) -> List[str]:
        """分割文本为多个 chunk"""
        return self._splitter.split_text(text)

    def create_chunks(
        self,
        text: str,
        doc_id: str,
        filename: str,
        metadata: dict = None
    ) -> List[dict]:
        """创建带元数据的 chunks"""
        text_chunks = self.split_text(text)
        chunks = []
        
        for idx, chunk_text in enumerate(text_chunks):
            chunk_data = {
                "content": chunk_text,
                "doc_id": doc_id,
                "chunk_index": idx,
                "filename": filename,
                "chunk_id": f"{doc_id}_{idx}",
                **{**(metadata or {})}
            }
            chunks.append(chunk_data)
        
        return chunks
