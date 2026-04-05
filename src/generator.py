"""
`generator.py` 是整个 RAG 系统中的“大模型生成层”，
负责把检索到的上下文交给 LLM 并生成最终回答。
它抽象了不同大模型提供商（如 OpenAI、Azure OpenAI），
通过统一的 `BaseGenerator` 接口实现可插拔式调用，同时支持同步、异步和流式输出。
文件中还包含 Prompt 模板管理（PromptTemplate / PromptManager），用于灵活构造提示词；
ContextBuilder 用于把检索结果拼接成符合 token 限制的上下文；以及 GeneratorFactory 工厂模式，用于根据配置动态创建不同生成器。
整体上，它的作用就是连接“检索结果”和“自然语言回答”，完成 RAG 的最后一步生成环节。


LLM Generation Module

该模块用于处理大模型（LLM）的文本生成能力，支持：

✔ 多种 LLM 提供商（OpenAI / Azure / 本地扩展）
✔ Prompt 模板管理
✔ 流式输出（Streaming）
✔ 上下文构建（RAG 专用）
"""

import os
from typing import List, Dict, Optional, AsyncIterator, Iterator, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from loguru import logger
import json


# ================================
# 1. 生成配置对象（控制 LLM 输出行为）
# ================================
@dataclass
class GenerationConfig:
    """
    控制生成行为的参数封装类

    类似 OpenAI API 中的参数封装：
    - temperature：控制随机性
    - max_tokens：最大输出长度
    - top_p：核采样
    - penalties：重复惩罚
    """
    temperature: float = 0.1
    max_tokens: int = 2000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None


# ================================
# 2. 生成结果封装
# ================================
@dataclass
class GenerationResult:
    """
    LLM 输出结果封装：

    - text：生成文本
    - tokens：用于计费/分析
    - metadata：扩展信息
    """
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ================================
# 3. 抽象生成器接口（多模型统一规范）
# ================================
class BaseGenerator(ABC):
    """
    所有 LLM Provider 的统一接口

    设计目的：
    ✔ 支持 OpenAI / Azure / 本地模型统一调用
    ✔ 解耦业务层与模型实现
    """

    @abstractmethod
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> GenerationResult:
        pass

    @abstractmethod
    async def generate_async(self, prompt: str, config: Optional[GenerationConfig] = None) -> GenerationResult:
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, config: Optional[GenerationConfig] = None) -> Iterator[str]:
        pass


# ================================
# 4. OpenAI 实现（核心生成器）
# ================================
class OpenAIGenerator(BaseGenerator):
    """
    OpenAI GPT 模型封装

    支持：
    ✔ 同步调用
    ✔ 异步调用
    ✔ 流式输出
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3
    ):
        # 延迟导入，避免未安装 openai 时整个项目崩溃
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OpenAI API key not provided")

        # 同步 client
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries
        )

        # 异步 client
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries
        )

        logger.info(f"Initialized OpenAI Generator (model={model})")


    # ================================
    # 4.1 普通生成（同步）
    # ================================
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> GenerationResult:
        config = config or GenerationConfig()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stop=config.stop_sequences
            )

            return GenerationResult(
                text=response.choices[0].message.content,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise


    # ================================
    # 4.2 异步生成
    # ================================
    async def generate_async(self, prompt: str, config: Optional[GenerationConfig] = None) -> GenerationResult:
        config = config or GenerationConfig()

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p
        )

        return GenerationResult(
            text=response.choices[0].message.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            finish_reason=response.choices[0].finish_reason
        )


    # ================================
    # 4.3 流式输出（边生成边返回）
    # ================================
    def generate_stream(self, prompt: str, config: Optional[GenerationConfig] = None) -> Iterator[str]:
        config = config or GenerationConfig()

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


    # ================================
    # 4.4 异步流式生成
    # ================================
    async def generate_stream_async(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:

        config = config or GenerationConfig()

        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ================================
# 5. Azure OpenAI（继承 OpenAI）
# ================================
class AzureOpenAIGenerator(OpenAIGenerator):
    """
    Azure OpenAI 封装版本

    差异：
    ✔ deployment_name 替代 model
    ✔ endpoint 需要 Azure 配置
    """

    def __init__(self, deployment_name: str, api_key=None, endpoint=None, api_version="2024-02-01"):
        from openai import AzureOpenAI, AsyncAzureOpenAI

        self.model = deployment_name

        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")

        if not api_key or not endpoint:
            raise ValueError("Azure OpenAI credentials not provided")

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )

        self.async_client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )


# ================================
# 6. Prompt 模板系统
# ================================
class PromptTemplate:
    """
    单个 Prompt 模板管理
    支持 {variable} 动态填充
    """

    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


class PromptManager:
    """
    管理多个 Prompt 模板，支持从 YAML 文件加载。

    用法：
        pm = PromptManager("config/prompts.yaml")
        tpl = pm.get("system.rag_assistant")
        prompt = tpl.format(context="...", question="...")
    """

    def __init__(self, prompts_file: Optional[str] = None):
        self.templates: Dict[str, PromptTemplate] = {}
        if prompts_file:
            self._load_yaml(prompts_file)

    def _load_yaml(self, path: str) -> None:
        """递归展开嵌套 YAML，用点号拼接 key，例如 system.rag_assistant"""
        import yaml
        if not os.path.exists(path):
            logger.warning(f"Prompts file not found: {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self._flatten(data, prefix="")

    def _flatten(self, obj, prefix: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}.{k}" if prefix else k
                self._flatten(v, new_key)
        elif isinstance(obj, str):
            self.templates[prefix] = PromptTemplate(obj)

    def add(self, key: str, template: str) -> None:
        self.templates[key] = PromptTemplate(template)

    def get(self, key: str) -> PromptTemplate:
        if key not in self.templates:
            raise KeyError(f"Prompt template '{key}' not found")
        return self.templates[key]


# ================================
# 7. RAG 上下文构建器（核心！）
# ================================
class ContextBuilder:
    """
    将 retrieval 的 chunks 拼接成 LLM prompt
    """

    def __init__(self, max_context_length: int = 4000, citation_format: str = "[[{index}]]"):
        self.max_context_length = max_context_length
        self.citation_format = citation_format

    def build(self, chunks: List[Dict]) -> str:
        context_parts = []
        total_length = 0

        for i, chunk in enumerate(chunks):
            citation = self.citation_format.format(index=i + 1)
            text = chunk.get("text", "")
            formatted = f"{citation} {text}"

            if total_length + len(formatted) > self.max_context_length:
                break

            context_parts.append(formatted)
            total_length += len(formatted)

        return "\n\n".join(context_parts)


# ================================
# 8. 工厂模式（创建生成器）
# ================================
class GeneratorFactory:
    """
    根据 config 创建不同 LLM provider
    """

    @staticmethod
    def create(config: Dict) -> BaseGenerator:
        provider = config.get("provider", "openai")

        if provider == "openai":
            return OpenAIGenerator(
                model=config.get("model", "gpt-4"),
                api_key=config.get("api_key"),
                base_url=config.get("base_url"),
                timeout=config.get("timeout", 60),
                max_retries=config.get("max_retries", 3)
            )

        elif provider == "azure":
            return AzureOpenAIGenerator(
                deployment_name=config.get("deployment_name"),
                api_key=config.get("api_key"),
                endpoint=config.get("endpoint")
            )

        else:
            raise ValueError("Unknown provider")