"""文本切片模块"""
from typing import List
import tiktoken
from loguru import logger
from app.config import settings


class RecursiveCharacterTextSplitter:
    """递归字符文本分割器"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", " ", ""]
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"无法加载 tiktoken 编码，使用字符数计算: {str(e)}")
            self.encoding = None

    def _count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            return len(text)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归分割文本"""
        if not text:
            return []

        if len(separators) == 0:
            return [text]

        separator = separators[0]
        new_separators = separators[1:]

        if separator == "":
            return [text]

        splits = text.split(separator)
        chunks = []
        current_chunk = ""

        for i, split in enumerate(splits):
            if i < len(splits) - 1:
                split = split + separator

            if not current_chunk:
                current_chunk = split
            elif self._count_tokens(current_chunk + split) <= self.chunk_size:
                current_chunk += split
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if self._count_tokens(split) > self.chunk_size:
                    sub_chunks = self._split_text(split, new_separators)
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = split

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def split_text(self, text: str) -> List[str]:
        """分割文本为多个 chunk"""
        if self._count_tokens(text) <= self.chunk_size:
            return [text]

        chunks = self._split_text(text, self.separators)

        if self.chunk_overlap > 0:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    overlapped_chunks.append(chunk)
                else:
                    prev_chunk = chunks[i - 1]
                    overlap_text = self._get_overlap_text(prev_chunk, self.chunk_overlap)
                    overlapped_chunks.append(overlap_text + chunk)
            return overlapped_chunks

        return chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """获取重叠文本"""
        tokens = self.encoding.encode(text) if self.encoding else list(text)
        if len(tokens) <= overlap_tokens:
            return text
        
        overlap_tokens_list = tokens[-overlap_tokens:]
        if self.encoding:
            return self.encoding.decode(overlap_tokens_list)
        else:
            return "".join(overlap_tokens_list)

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
                "metadata": metadata or {}
            }
            chunks.append(chunk_data)
        
        return chunks

