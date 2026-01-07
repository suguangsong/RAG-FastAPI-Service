"""LLM 服务模块"""
from typing import List, Dict, Any, Optional, AsyncIterator
from loguru import logger
from app.config import settings


class LLMService:
    """LLM 服务"""

    def __init__(self):
        self.provider = settings.llm_provider
        self.model_name = settings.llm_model_name
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """初始化 LLM 客户端"""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=settings.openai_api_key)
            except Exception as e:
                logger.error(f"OpenAI 客户端初始化失败: {str(e)}")
                raise
        elif self.provider == "dashscope":
            try:
                import dashscope
                dashscope.api_key = settings.dashscope_api_key
                self._client = dashscope
            except Exception as e:
                logger.error(f"DashScope 客户端初始化失败: {str(e)}")
                raise
        elif self.provider == "ollama":
            self._client = None
        else:
            raise ValueError(f"不支持的 LLM 提供商: {self.provider}")

    def _build_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """构建 RAG 提示词"""
        context_text = "\n\n".join([
            f"[文档片段 {idx + 1}]:\n{doc['content']}"
            for idx, doc in enumerate(context)
        ])
        
        prompt = f"""基于以下上下文信息回答用户问题。如果上下文中没有相关信息，请说明无法从提供的文档中找到答案。

上下文信息：
{context_text}

用户问题：{query}

请提供准确、简洁的回答："""
        return prompt

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
            return await self._generate_stream(query, context, temperature, max_tokens)
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
        prompt = self._build_prompt(query, context)
        temp = temperature if temperature is not None else self.temperature
        max_toks = max_tokens if max_tokens is not None else self.max_tokens
        
        if self.provider == "openai":
            return await self._generate_openai(prompt, temp, max_toks)
        elif self.provider == "dashscope":
            return await self._generate_dashscope(prompt, temp, max_toks)
        elif self.provider == "ollama":
            return await self._generate_ollama(prompt, temp, max_toks)
        else:
            raise ValueError(f"不支持的 LLM 提供商: {self.provider}")

    async def _generate_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """使用 OpenAI 生成"""
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return {
                "answer": answer,
                "usage": usage
            }
        except Exception as e:
            logger.error(f"OpenAI 生成失败: {str(e)}")
            raise

    async def _generate_dashscope(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """使用 DashScope 生成"""
        try:
            import dashscope
            response = dashscope.Generation.call(
                model=self.model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if response.status_code == 200:
                answer = response.output.choices[0].message.content
                usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
                return {
                    "answer": answer,
                    "usage": usage
                }
            else:
                raise ValueError(f"DashScope API 错误: {response.message}")
        except Exception as e:
            logger.error(f"DashScope 生成失败: {str(e)}")
            raise

    async def _generate_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """使用 Ollama 生成"""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json={
                        "model": settings.ollama_llm_model,
                        "prompt": prompt,
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "stream": False
                    },
                    timeout=120.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("response", "")
                    return {
                        "answer": answer,
                        "usage": None
                    }
                else:
                    raise ValueError(f"Ollama API 错误: {response.status_code}")
        except Exception as e:
            logger.error(f"Ollama 生成失败: {str(e)}")
            raise

    async def _generate_stream(
        self,
        query: str,
        context: List[Dict[str, Any]],
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> AsyncIterator[str]:
        """流式生成"""
        prompt = self._build_prompt(query, context)
        temp = temperature if temperature is not None else self.temperature
        max_toks = max_tokens if max_tokens is not None else self.max_tokens
        
        if self.provider == "openai":
            async for chunk in self._generate_openai_stream(prompt, temp, max_toks):
                yield chunk
        elif self.provider == "ollama":
            async for chunk in self._generate_ollama_stream(prompt, temp, max_toks):
                yield chunk
        else:
            raise ValueError(f"流式生成暂不支持提供商: {self.provider}")

    async def _generate_openai_stream(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> AsyncIterator[str]:
        """OpenAI 流式生成"""
        try:
            stream = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI 流式生成失败: {str(e)}")
            raise

    async def _generate_ollama_stream(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> AsyncIterator[str]:
        """Ollama 流式生成"""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{settings.ollama_base_url}/api/generate",
                    json={
                        "model": settings.ollama_llm_model,
                        "prompt": prompt,
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "stream": True
                    },
                    timeout=120.0
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            import json
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.error(f"Ollama 流式生成失败: {str(e)}")
            raise

