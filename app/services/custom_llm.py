"""自定义 LLM 实现"""
from typing import List, Optional, Iterator, AsyncIterator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from loguru import logger
import zhipuai
import dashscope


class CustomChatZhipuAI(BaseChatModel):
    """自定义智谱AI Chat Model"""

    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    zhipuai_api_key: str

    def __init__(self, model: str, temperature: float, max_tokens: int, zhipuai_api_key: str):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.zhipuai_api_key = zhipuai_api_key
        self._client = zhipuai.ZhipuAI(api_key=zhipuai_api_key)

    @property
    def _llm_type(self) -> str:
        return "zhipuai"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        import asyncio
        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        try:
            # 转换消息格式
            formatted_messages = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    formatted_messages.append({
                        "role": msg.__class__.__name__.lower().replace("message", "").replace("human", "user"),
                        "content": msg.content
                    })
            
            response = self._client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            message = response.choices[0].message
            generation = ChatGeneration(
                message=type(messages[0])(content=message.content),
                generation_info=response.usage.model_dump() if hasattr(response, 'usage') else {}
            )
            
            return ChatResult(generations=[generation])
        except Exception as e:
            logger.error(f"智谱AI 生成失败: {str(e)}")
            raise

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs
    ) -> AsyncIterator:
        try:
            formatted_messages = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    formatted_messages.append({
                        "role": msg.__class__.__name__.lower().replace("message", "").replace("human", "user"),
                        "content": msg.content
                    })
            
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield type(messages[0])(content=delta.content)
        except Exception as e:
            logger.error(f"智谱AI 流式生成失败: {str(e)}")
            raise


class CustomChatTongyi(BaseChatModel):
    """自定义 DashScope Chat Model"""

    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    dashscope_api_key: str

    def __init__(self, model: str, temperature: float, max_tokens: int, dashscope_api_key: str):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dashscope_api_key = dashscope_api_key
        dashscope.api_key = dashscope_api_key

    @property
    def _llm_type(self) -> str:
        return "dashscope"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        import asyncio
        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        try:
            import dashscope
            prompt = "\n".join([msg.content for msg in messages if hasattr(msg, 'content')])
            
            response = dashscope.Generation.call(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if response.status_code == 200:
                answer = response.output.choices[0].message.content
                generation = ChatGeneration(
                    message=type(messages[0])(content=answer),
                    generation_info=response.usage.model_dump() if hasattr(response, 'usage') else {}
                )
                return ChatResult(generations=[generation])
            else:
                raise ValueError(f"DashScope API 错误: {response.message}")
        except Exception as e:
            logger.error(f"DashScope 生成失败: {str(e)}")
            raise
