"""LLM 服务模块 - 使用 LangChain"""
from typing import List, Dict, Any, Optional, AsyncIterator
from loguru import logger
from app.config import settings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
try:
    from langchain_community.chat_models import ChatZhipuAI, ChatTongyi, ChatOllama
except ImportError:
    # 如果导入失败，使用自定义实现
    ChatZhipuAI = None
    ChatTongyi = None
    from langchain_community.chat_models import ChatOllama


class LLMService:
    """LLM 服务 - 基于 LangChain"""

    def __init__(self):
        self.provider = settings.llm_provider
        self.model_name = settings.llm_model_name
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self._llm: Optional[BaseChatModel] = None
        self._initialize_llm()

    def _initialize_llm(self):
        """初始化 LangChain LLM"""
        try:
            if self.provider == "zhipuai":
                if ChatZhipuAI is None:
                    # 使用自定义实现
                    from app.services.custom_llm import CustomChatZhipuAI
                    self._llm = CustomChatZhipuAI(
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        zhipuai_api_key=settings.zhipuai_api_key
                    )
                else:
                    self._llm = ChatZhipuAI(
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        zhipuai_api_key=settings.zhipuai_api_key
                    )
            elif self.provider == "openai":
                self._llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    openai_api_key=settings.openai_api_key
                )
            elif self.provider == "dashscope":
                if ChatTongyi is None:
                    # 使用自定义实现
                    from app.services.custom_llm import CustomChatTongyi
                    self._llm = CustomChatTongyi(
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        dashscope_api_key=settings.dashscope_api_key
                    )
                else:
                    self._llm = ChatTongyi(
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        dashscope_api_key=settings.dashscope_api_key
                    )
            elif self.provider == "ollama":
                self._llm = ChatOllama(
                    model=settings.ollama_llm_model,
                    temperature=self.temperature,
                    num_predict=self.max_tokens,
                    base_url=settings.ollama_base_url
                )
            else:
                raise ValueError(f"不支持的 LLM 提供商: {self.provider}")
            logger.info(f"LLM 初始化成功: {self.provider}/{self.model_name}")
        except Exception as e:
            logger.error(f"LLM 初始化失败: {str(e)}")
            raise

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """构建 RAG 提示词模板"""
        return ChatPromptTemplate.from_messages([
            ("system", """基于以下上下文信息回答用户问题。如果上下文中没有相关信息，请说明无法从提供的文档中找到答案。

上下文信息：
{context}

请提供准确、简洁的回答："""),
            ("human", "{query}")
        ])

    async def generate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """生成回答"""
        if stream:
            return self._generate_stream(query, context, temperature, max_tokens)
        else:
            return await self._generate_non_stream(query, context, temperature, max_tokens)

    async def _generate_non_stream(
        self,
        query: str,
        context: List[Dict[str, Any]],
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """非流式生成"""
        try:
            # 构建上下文文本
            context_text = "\n\n".join([
                f"[文档片段 {idx + 1}]:\n{doc['content']}"
                for idx, doc in enumerate(context)
            ])
            
            # 构建提示词
            prompt_template = self._build_prompt_template()
            messages = prompt_template.format_messages(
                context=context_text,
                query=query
            )
            
            # 临时设置参数（如果提供）
            llm = self._llm
            if temperature is not None or max_tokens is not None:
                llm = self._llm.bind(
                    temperature=temperature if temperature is not None else self.temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.max_tokens
                )
            
            # 调用 LLM
            response = await llm.ainvoke(messages)
            answer = response.content
            
            # 获取使用信息
            usage = None
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if 'token_usage' in metadata:
                    token_usage = metadata['token_usage']
                    usage = {
                        "prompt_tokens": token_usage.get('prompt_tokens', 0),
                        "completion_tokens": token_usage.get('completion_tokens', 0),
                        "total_tokens": token_usage.get('total_tokens', 0)
                    }
            
            return {
                "answer": answer,
                "usage": usage
            }
        except Exception as e:
            logger.error(f"LLM 生成失败: {str(e)}")
            raise

    async def _generate_stream(
        self,
        query: str,
        context: List[Dict[str, Any]],
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> AsyncIterator[str]:
        """流式生成"""
        try:
            # 构建上下文文本
            context_text = "\n\n".join([
                f"[文档片段 {idx + 1}]:\n{doc['content']}"
                for idx, doc in enumerate(context)
            ])
            
            # 构建提示词
            prompt_template = self._build_prompt_template()
            messages = prompt_template.format_messages(
                context=context_text,
                query=query
            )
            
            # 临时设置参数（如果提供）
            llm = self._llm
            if temperature is not None or max_tokens is not None:
                llm = self._llm.bind(
                    temperature=temperature if temperature is not None else self.temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.max_tokens
                )
            
            # 流式调用
            async for chunk in llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"LLM 流式生成失败: {str(e)}")
            raise
