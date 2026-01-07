"""文档解析模块 - 使用 LangChain"""
import io
from typing import List
from pathlib import Path
from loguru import logger
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document


class DocumentParser:
    """文档解析器 - 基于 LangChain"""

    @staticmethod
    def parse_pdf(file_content: bytes, filename: str) -> str:
        """解析 PDF 文件"""
        try:
            # 创建临时文件对象
            pdf_file = io.BytesIO(file_content)
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            text = "\n".join([doc.page_content for doc in documents])
            return text.strip()
        except Exception as e:
            logger.error(f"PDF 解析失败: {filename}, 错误: {str(e)}")
            raise ValueError(f"PDF 解析失败: {str(e)}")

    @staticmethod
    def parse_docx(file_content: bytes, filename: str) -> str:
        """解析 DOCX 文件"""
        try:
            # 创建临时文件对象
            docx_file = io.BytesIO(file_content)
            loader = Docx2txtLoader(docx_file)
            documents = loader.load()
            text = "\n".join([doc.page_content for doc in documents])
            return text.strip()
        except Exception as e:
            logger.error(f"DOCX 解析失败: {filename}, 错误: {str(e)}")
            raise ValueError(f"DOCX 解析失败: {str(e)}")

    @staticmethod
    def parse_txt(file_content: bytes, filename: str) -> str:
        """解析 TXT 文件"""
        try:
            # 创建临时文件对象
            txt_file = io.BytesIO(file_content)
            loader = TextLoader(txt_file, encoding='utf-8')
            documents = loader.load()
            text = "\n".join([doc.page_content for doc in documents])
            return text.strip()
        except UnicodeDecodeError:
            try:
                txt_file = io.BytesIO(file_content)
                loader = TextLoader(txt_file, encoding='gbk')
                documents = loader.load()
                text = "\n".join([doc.page_content for doc in documents])
                return text.strip()
            except Exception as e:
                logger.error(f"TXT 解析失败: {filename}, 错误: {str(e)}")
                raise ValueError(f"TXT 解析失败: 无法解码文件内容")
        except Exception as e:
            logger.error(f"TXT 解析失败: {filename}, 错误: {str(e)}")
            raise ValueError(f"TXT 解析失败: {str(e)}")

    @classmethod
    def parse(cls, file_content: bytes, filename: str) -> str:
        """根据文件扩展名自动选择解析方法"""
        file_ext = Path(filename).suffix.lower()
        
        if file_ext == '.pdf':
            return cls.parse_pdf(file_content, filename)
        elif file_ext in ['.docx', '.doc']:
            return cls.parse_docx(file_content, filename)
        elif file_ext == '.txt':
            return cls.parse_txt(file_content, filename)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
