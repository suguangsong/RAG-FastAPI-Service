"""文档解析模块"""
import io
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from docx import Document
from loguru import logger


class DocumentParser:
    """文档解析器"""

    @staticmethod
    def parse_pdf(file_content: bytes, filename: str) -> str:
        """解析 PDF 文件"""
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PDF 解析失败: {filename}, 错误: {str(e)}")
            raise ValueError(f"PDF 解析失败: {str(e)}")

    @staticmethod
    def parse_docx(file_content: bytes, filename: str) -> str:
        """解析 DOCX 文件"""
        try:
            docx_file = io.BytesIO(file_content)
            doc = Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"DOCX 解析失败: {filename}, 错误: {str(e)}")
            raise ValueError(f"DOCX 解析失败: {str(e)}")

    @staticmethod
    def parse_txt(file_content: bytes, filename: str) -> str:
        """解析 TXT 文件"""
        try:
            text = file_content.decode('utf-8')
            return text.strip()
        except UnicodeDecodeError:
            try:
                text = file_content.decode('gbk')
                return text.strip()
            except Exception as e:
                logger.error(f"TXT 解析失败: {filename}, 错误: {str(e)}")
                raise ValueError(f"TXT 解析失败: 无法解码文件内容")

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

