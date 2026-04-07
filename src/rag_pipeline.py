"""
这个 rag_pipeline.py 是整个系统的核心调度器（Orchestrator），
负责把“文档 → 切分 → 向量化 → 检索 → Prompt 构造 → LLM 生成 →（可选评估）”完整串联成一条生产级 RAG 流水线，是整个项目真正运行的主入口。
RAG Pipeline Orchestration（RAG主流程编排模块）

本文件是整个 RAG 系统的“核心控制器”，负责把所有模块串起来：

📌 主要流程：
1. 加载文档（DataLoader）
2. 文档切分（Splitter）
3. 向量化（Embedding Model）
4. 存入向量数据库（Vector Store）
5. 检索相关内容（Retriever）
6. 构造上下文（Context Builder）
7. 大模型生成回答（Generator）
8. 可选：评估（Evaluator）

👉 本质：RAG 的“生产级流水线引擎”
"""

import os
import yaml
import json
from typing import List, Dict, Optional, Union, Iterator, AsyncIterator
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

# =========================
# 数据加载模块
# =========================
from .data_loader import DataLoader, Document

# =========================
# 文档切分模块
# =========================
from .document_splitter import (
    RecursiveCharacterSplitter,
    TextChunk,
    SplitterFactory
)

# =========================
# 向量化模型
# =========================
from .embedding_model import (
    EmbeddingModelFactory,
    BaseEmbeddingModel
)

# =========================
# 向量数据库
# =========================
from .vector_store import (
    VectorStoreFactory,
    BaseVectorStore,
    VectorRecord
)

# =========================
# 检索器（混合检索 / 重排序）
# =========================
from .retriever import AdvancedRetriever

# =========================
# LLM 生成模块
# =========================
from .generator import (
    GeneratorFactory,
    PromptManager,
    ContextBuilder,
    GenerationConfig,
    GenerationResult
)

# =========================
# 评估模块（可选）
# =========================
from .evaluation import RAGEvaluator


# =========================================================
# 对话轮次结构
# =========================================================
@dataclass
class ConversationTurn:
    """单轮对话记录"""
    question: str
    answer: str
    turn_id: int


# =========================================================
# 对话历史管理器（长上下文压缩核心）
# =========================================================
class ConversationManager:
    """
    多轮对话历史管理 + 上下文压缩

    策略：
    - 保留最近 keep_recent 轮完整对话
    - 更早的轮次用 LLM 压缩成摘要
    - 总 token 超限时从最早的摘要开始丢弃
    """

    def __init__(self, keep_recent: int = 2, max_history_tokens: int = 1000):
        self.history: List[ConversationTurn] = []
        self.keep_recent = keep_recent          # 保留完整的最近N轮
        self.max_history_tokens = max_history_tokens
        self.summary: str = ""                  # 压缩后的历史摘要

    def add_turn(self, question: str, answer: str) -> None:
        turn = ConversationTurn(
            question=question,
            answer=answer,
            turn_id=len(self.history) + 1
        )
        self.history.append(turn)

    def get_recent_turns(self) -> List[ConversationTurn]:
        """获取最近 keep_recent 轮完整对话"""
        return self.history[-self.keep_recent:] if self.history else []

    def get_older_turns(self) -> List[ConversationTurn]:
        """获取需要压缩的较早轮次"""
        if len(self.history) <= self.keep_recent:
            return []
        return self.history[:-self.keep_recent]

    def format_recent_history(self) -> str:
        """格式化最近几轮对话为字符串"""
        turns = self.get_recent_turns()
        if not turns:
            return ""
        lines = []
        for t in turns:
            lines.append(f"用户[第{t.turn_id}轮]：{t.question}")
            lines.append(f"助手[第{t.turn_id}轮]：{t.answer[:300]}...")
        return "\n".join(lines)

    def compress_history(self, generator, gen_config) -> str:
        """
        增量压缩：只压缩上次摘要之后新增的旧轮次，避免重复压缩已处理的历史
        """
        older = self.get_older_turns()
        if not older:
            return self.summary

        # 已经压缩过的轮次数（通过 summary 对应的轮次追踪）
        already_compressed = len(older) - (len(older) - getattr(self, '_last_compressed_count', 0))
        new_older = older[getattr(self, '_last_compressed_count', 0):]

        if not new_older:
            # 没有新增的旧轮次，无需重新压缩
            return self.summary

        # 只压缩新增的旧轮次，结合已有摘要
        new_history_text = "\n".join([
            f"用户：{t.question}\n助手：{t.answer[:500]}"
            for t in new_older
        ])

        if self.summary:
            # 在已有摘要基础上追加新内容
            compress_prompt = f"""请在已有摘要基础上，将新增对话内容合并进去，生成更新后的摘要（不超过150字）。
重点保留：用户关注的核心问题、已确认的关键事实、未解决的疑问。
丢弃：重复信息、已解决的细节。

已有摘要：
{self.summary}

新增对话：
{new_history_text}

更新后的摘要："""
        else:
            compress_prompt = f"""请将以下对话历史压缩为一段简短摘要（不超过150字）。
重点保留：用户关注的核心问题、已确认的关键事实、未解决的疑问。
丢弃：重复信息、寒暄、已解决的细节。

对话历史：
{new_history_text}

摘要："""

        try:
            result = generator.generate(compress_prompt, gen_config)
            self.summary = result.text.strip()
            self._last_compressed_count = len(older)  # 记录已压缩到第几轮
            logger.info(f"History compressed: {len(new_history_text)} chars -> {len(self.summary)} chars")
        except Exception as e:
            logger.warning(f"History compression failed: {e}")
            self.summary = (self.summary + "\n" + new_history_text)[:500] + "..."
            self._last_compressed_count = len(older)

        return self.summary

    def clear(self) -> None:
        self.history = []
        self.summary = ""
        self._last_compressed_count = 0


# =========================================================
# RAG 输出统一结构
# =========================================================
@dataclass
class RAGResponse:
    """
    RAG 最终返回结构（API 层直接使用）

    Attributes:
        answer: 最终生成答案
        sources: 引用来源（chunk信息）
        metadata: 运行信息（token、检索数等）
    """
    answer: str
    sources: List[Dict]
    metadata: Dict

    def to_dict(self) -> Dict:
        """转 JSON 输出"""
        return {
            'answer': self.answer,
            'sources': self.sources,
            'metadata': self.metadata
        }


# =========================================================
# RAG 主流水线
# =========================================================
class RAGPipeline:
    """
    生产级 RAG Pipeline（核心类）

    使用方式：
        pipeline = RAGPipeline.from_config("config.yaml")
        pipeline.index_documents("data/")
        result = pipeline.query("什么是RAG？")
    """

    def __init__(
        self,
        config: Dict,
        data_loader: Optional[DataLoader] = None,
        splitter=None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        vector_store: Optional[BaseVectorStore] = None,
        retriever=None,
        generator=None,
        prompt_manager: Optional[PromptManager] = None
    ):
        self.config = config

        # =========================
        # 1. 文档加载器
        # =========================
        self.data_loader = data_loader or DataLoader(config.get('document', {}))

        # =========================
        # 2. 文档切分器
        # =========================
        if splitter is None:
            self.splitter = RecursiveCharacterSplitter(
                chunk_size=config.get('document', {}).get('default_chunk_size', 512),
                chunk_overlap=config.get('document', {}).get('default_chunk_overlap', 50)
            )
        else:
            self.splitter = splitter

        # =========================
        # 3. Embedding 模型
        # =========================
        self.embedding_model = embedding_model or EmbeddingModelFactory.create(
            config.get('embedding', {})
        )

        # =========================
        # 4. 向量数据库
        # =========================
        self.vector_store = vector_store or VectorStoreFactory.create(
            config.get('vector_store', {})
        )

        # =========================
        # 5. 检索器（Rerank + Hybrid Search）
        # =========================
        if retriever is None:
            self.retriever = AdvancedRetriever(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                config=config.get('retrieval', {})
            )
        else:
            self.retriever = retriever

        # =========================
        # 6. LLM 生成器
        # =========================
        self.generator = generator or GeneratorFactory.create(
            config.get('llm', {})
        )

        # =========================
        # 7. Prompt 管理器
        # =========================
        prompts_file = os.path.join('config', 'prompts.yaml')
        self.prompt_manager = prompt_manager or PromptManager(prompts_file)

        # =========================
        # 8. 上下文构造器（控制token长度）
        # =========================
        self.context_builder = ContextBuilder(
            max_context_length=config.get('pipeline', {}).get('max_context_length', 4000),
            citation_format=config.get('pipeline', {}).get('citation_format', '[[{index}]]')
        )

        # =========================
        # 9. 评估器（可选）
        # =========================
        self.evaluator = None
        if config.get('evaluation', {}).get('enabled', False):
            self.evaluator = RAGEvaluator()

        # =========================
        # 10. 对话历史管理器
        # =========================
        pipeline_cfg = config.get('pipeline', {})
        self.conversation = ConversationManager(
            keep_recent=pipeline_cfg.get('keep_recent_turns', 2),
            max_history_tokens=pipeline_cfg.get('max_history_tokens', 1000)
        )

        # Token 预算（8K窗口分层分配）
        self.max_total_tokens = pipeline_cfg.get('max_total_tokens', 7000)
        self.token_budget = {
            'system_prompt': 500,    # 系统提示固定占用
            'history': 1000,         # 对话历史（压缩后）
            'context': 4000,         # 检索文档
            'question': 200,         # 用户问题
            'answer': 1500,          # 留给回答的空间
        }

        logger.info("RAG Pipeline initialized successfully")

    # =========================================================
    # 从 YAML 配置初始化
    # =========================================================
    @classmethod
    def from_config(cls, config_path: str) -> 'RAGPipeline':
        """从配置文件创建 Pipeline"""
        from dotenv import load_dotenv
        load_dotenv()

        # 设置代理（如果 .env 中配置了）
        import os
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        if https_proxy:
            os.environ["HTTPS_PROXY"] = https_proxy
            os.environ["HTTP_PROXY"] = https_proxy

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return cls(cls._expand_env_vars(config))

    # =========================================================
    # 环境变量解析（${VAR}）
    # =========================================================
    @staticmethod
    def _expand_env_vars(obj):
        """
        递归解析配置中的环境变量：
        ${OPENAI_API_KEY}
        ${VAR:default}
        """
        if isinstance(obj, dict):
            return {k: RAGPipeline._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [RAGPipeline._expand_env_vars(i) for i in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            default = None
            if ':' in env_var:
                env_var, default = env_var.split(':', 1)
            return os.getenv(env_var, default)
        return obj

    # =========================================================
    # 文档索引（核心 ingestion 流程）
    # =========================================================
    def index_documents(
        self,
        source: Union[str, Path, List[Document]],
        batch_size: int = 100
    ) -> None:
        """
        将文档 → 切分 → embedding → 写入向量库
        """

        logger.info(f"Indexing source: {source}")

        # -------- 1. 加载文档 --------
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                documents = self.data_loader.load_directory(path)
            else:
                documents = [self.data_loader.load_document(path)]
        else:
            documents = source

        if not documents:
            logger.warning("No documents found")
            return

        logger.info(f"Loaded {len(documents)} documents")

        # -------- 2. 文档切分 --------
        all_chunks = self.splitter.split_batch(documents)
        logger.info(f"Generated {len(all_chunks)} chunks")

        # -------- 3. 分批 embedding + 入库 --------
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]

            texts = [c.content for c in batch]
            embeddings = self.embedding_model.embed_documents(texts)

            records = []
            for chunk, emb in zip(batch, embeddings):
                records.append(VectorRecord(
                    id=chunk.chunk_id,
                    vector=emb,
                    text=chunk.content,
                    metadata=chunk.metadata
                ))

            self.vector_store.add(records)

        logger.info("Indexing completed")

        # -------- 4. 构建关键词索引并持久化 --------
        # 从向量库读取所有 chunk，重建关键词倒排索引
        # 解决：关键词索引未初始化导致混合检索退化为纯向量检索的问题
        try:
            all_results = self.vector_store.collection.get()
            keyword_docs = [
                {"id": all_results["ids"][i], "text": all_results["documents"][i]}
                for i in range(len(all_results["ids"]))
            ]
            self.retriever.hybrid.build_keyword_index(keyword_docs)
            self.retriever.hybrid.save_index()
        except Exception as e:
            logger.warning(f"Keyword index build failed: {e}")

    # =========================================================
    # 查询入口（RAG 主逻辑）
    # =========================================================
    def query(
        self,
        question: str,
        top_k: int = 5,
        stream: bool = False,
        use_history: bool = True
    ) -> Union[RAGResponse, Iterator[str]]:

        # -------- 1. 检索 --------
        # 优先级 4：多轮对话查询改写，消除指代词歧义
        retrieval_question = question
        if use_history and self.conversation.history:
            retrieval_question = self._rewrite_question(question)

        retrieved_chunks = self.retriever.retrieve(retrieval_question, top_k=top_k)

        if not retrieved_chunks:
            return RAGResponse(
                answer="No relevant information found.",
                sources=[],
                metadata={'retrieved_count': 0}
            )

        # -------- 2. 相似度阈值过滤（去除噪声chunk）--------
        threshold = self.config.get('retrieval', {}).get('similarity_threshold', 0.3)
        filtered_chunks = [c for c in retrieved_chunks if c.final_score >= threshold]

        # 无关问题检测：只有在完全没有 chunk 通过阈值时才拒答
        # 不依赖 final_score 绝对值（混合检索归一化后分数普遍偏低），
        # 改为：若最高分 chunk 的 vector_score 低于 off_topic_threshold 则判定无关
        off_topic_threshold = self.config.get('retrieval', {}).get('off_topic_threshold', 0.3)
        top_vector_score = retrieved_chunks[0].vector_score if retrieved_chunks else 0
        if not filtered_chunks and top_vector_score < off_topic_threshold:
            return RAGResponse(
                answer="抱歉，您的问题超出了我的知识范围，我只能回答与文档相关的技术问题。",
                sources=[],
                metadata={'retrieved_count': 0, 'reason': 'off_topic'}
            )

        if not filtered_chunks:
            filtered_chunks = retrieved_chunks[:3]  # 至少保留top3

        # -------- 3. 构造检索上下文（token预算控制）--------
        context = self._build_context_with_budget(filtered_chunks)

        # -------- 4. 构造历史摘要（多轮对话压缩）--------
        history_text = ""
        if use_history and self.conversation.history:
            gen_config = GenerationConfig(
                temperature=0.1,
                max_tokens=300
            )
            # 压缩较早的历史
            if len(self.conversation.history) > self.conversation.keep_recent:
                self.conversation.compress_history(self.generator, gen_config)

            history_summary = self.conversation.summary
            recent = self.conversation.format_recent_history()

            parts = []
            if history_summary:
                parts.append(f"【早期对话摘要】\n{history_summary}")
            if recent:
                parts.append(f"【最近对话】\n{recent}")
            history_text = "\n\n".join(parts)

        # -------- 5. 选择 Prompt 模板 --------
        try:
            if history_text:
                template = self.prompt_manager.get('system.conversational')
            else:
                template = self.prompt_manager.get('system.rag_assistant')
        except KeyError:
            from .generator import PromptTemplate
            template = PromptTemplate(
                "根据以下上下文回答问题。\n上下文：{context}\n问题：{question}\n回答："
            )

        # -------- 6. 格式化 Prompt --------
        try:
            if history_text:
                prompt = template.format(
                    context=context,
                    question=question,
                    history_summary=history_text
                )
            else:
                prompt = template.format(context=context, question=question)
        except KeyError:
            prompt = f"上下文：{context}\n\n问题：{question}\n\n回答："

        # 优化四：Prompt 超限兜底——丢弃历史，只保留 context + question
        prompt_tokens = self._count_tokens(prompt)
        if prompt_tokens > self.max_total_tokens:
            logger.warning(f"Prompt too long ({prompt_tokens} tokens), dropping history")
            try:
                fallback_template = self.prompt_manager.get('system.rag_assistant')
                prompt = fallback_template.format(context=context, question=question)
            except Exception:
                prompt = f"上下文：{context}\n\n问题：{question}\n\n回答："

        # -------- 7. 生成答案 --------
        if stream:
            return self._stream_response(prompt, filtered_chunks, question)
        else:
            response = self._generate_response(prompt, filtered_chunks, question)
            # 保存本轮到历史
            if use_history:
                self.conversation.add_turn(question, response.answer)
            return response

    # =========================================================
    # 分层上下文构建（token预算控制）
    # =========================================================
    def _build_context_with_budget(self, chunks) -> str:
        """
        在 token 预算内构建上下文

        优化：
        1. 不截断单个 chunk（整体丢弃低相关 chunk，保证语义完整性）
        2. 精确 token 计数（tiktoken，降级为字符估算）
        3. 总长度不超过 context token 预算
        """
        budget_tokens = self.token_budget['context']
        context_parts = []
        used_tokens = 0

        for i, chunk in enumerate(chunks):
            citation = f"[[{i+1}]]"
            text = chunk.text

            # chunk 格式化加入文档名，帮助 LLM 感知内容来源
            filename = chunk.metadata.get('filename', '') if chunk.metadata else ''
            section_path = chunk.metadata.get('section_path', '') if chunk.metadata else ''
            header = ' | '.join(filter(None, [filename, section_path]))
            if header:
                text = f"[{header}]\n{text}"

            formatted = f"{citation} {text}"
            chunk_tokens = self._count_tokens(formatted)

            # 优化一：超出预算时整体跳过，不截断单个 chunk
            if used_tokens + chunk_tokens > budget_tokens:
                break

            context_parts.append(formatted)
            used_tokens += chunk_tokens

        return "\n\n".join(context_parts)

    @staticmethod
    def _count_tokens(text: str) -> int:
        """
        优化二：精确 token 计数，降级为字符估算。
        tiktoken 未安装时自动降级（1 token ≈ 2.5 字符，中英混合更准确）。
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            return len(text) // 2  # 中英混合文本，2.5 字符/token 更接近实际

    # =========================================================
    # 非流式生成
    # =========================================================
    def _generate_response(self, prompt, chunks, question) -> RAGResponse:

        # 优化三：动态生成预算——确保输入 + 输出不超过模型上下文窗口
        input_tokens = self._count_tokens(prompt)
        max_window = self.max_total_tokens
        available_for_output = max(300, max_window - input_tokens)
        configured_max = self.config.get('llm', {}).get('max_tokens', 2000)

        config = GenerationConfig(
            temperature=self.config.get('llm', {}).get('temperature', 0.1),
            max_tokens=min(configured_max, available_for_output)
        )

        result = self.generator.generate(prompt, config)

        # 优先级 1：引用验证——过滤超出范围的引用序号，防止引用幻觉
        answer_text = self._fix_citations(result.text, len(chunks))

        sources = [
            {
                "index": i + 1,
                "text": c.text[:200],
                "metadata": c.metadata,
                "score": c.final_score,
                "section_path": c.metadata.get('section_path', '') if c.metadata else '',
            }
            for i, c in enumerate(chunks)
        ]

        evaluation = None
        if self.evaluator:
            evaluation = self.evaluator.evaluate(
                question=question,
                answer=answer_text,
                contexts=[c.text for c in chunks]
            )

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            metadata={
                "retrieved_count": len(chunks),
                "tokens": result.total_tokens,
                "evaluation": evaluation
            }
        )

    # =========================================================
    # 流式生成
    # =========================================================
    def _stream_response(self, prompt, chunks, question):

        config = GenerationConfig(
            temperature=self.config.get('llm', {}).get('temperature', 0.1),
            max_tokens=self.config.get('llm', {}).get('max_tokens', 2000)
        )

        # 先返回 sources
        sources = [
            {
                "index": i + 1,
                "score": c.final_score
            }
            for i, c in enumerate(chunks)
        ]

        yield f"__SOURCES__:{json.dumps(sources)}\n"

        # 再流式输出答案
        for token in self.generator.generate_stream(prompt, config):
            yield token

    # =========================================================
    # 优先级 1：引用验证（过滤超出范围的引用序号）
    # =========================================================
    @staticmethod
    def _fix_citations(text: str, max_idx: int) -> str:
        """
        过滤答案中超出 chunk 范围的引用序号。
        例如只有 5 个 chunk，但 LLM 生成了 [[6]]，则删除该引用。
        """
        import re
        def replace(m):
            idx = int(m.group(1))
            return m.group(0) if 1 <= idx <= max_idx else ''
        return re.sub(r'\[\[(\d+)\]\]', replace, text)

    # =========================================================
    # 优先级 4：多轮对话查询改写（消除指代词歧义）
    # =========================================================
    def _rewrite_question(self, question: str) -> str:
        """
        当有对话历史时，用 LLM 把含指代词的问题改写为完整问题。
        例如："它的参数是多少？" → "液压泵 HPV-135 的参数是多少？"
        改写失败时返回原始问题，不影响主流程。
        """
        recent = self.conversation.format_recent_history()
        if not recent:
            return question

        # 简单判断：问题里有指代词才改写，避免无谓的 LLM 调用
        pronouns = ['它', '这个', '该', '上述', '前面', '之前', '其', '这些', '那个']
        if not any(p in question for p in pronouns):
            return question

        prompt = (
            f"历史对话：\n{recent}\n\n"
            f"当前问题：{question}\n\n"
            f"如果当前问题包含指代词（它、这个、该、上述等），请将其替换为具体内容，改写为完整问题；"
            f"如果问题已经完整，原样返回。只输出改写后的问题，不要解释。\n"
            f"改写后的问题："
        )
        try:
            result = self.generator.generate(
                prompt,
                GenerationConfig(temperature=0, max_tokens=100)
            )
            rewritten = result.text.strip()
            if rewritten and len(rewritten) < len(question) * 3:  # 防止改写过长
                logger.debug(f"Query rewritten: '{question}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.debug(f"Query rewrite failed: {e}")
        return question

    # =========================================================
    # 统计信息
    # =========================================================
    def get_stats(self) -> Dict:
        return {
            "vector_count": self.vector_store.count(),
            "embedding_dim": self.embedding_model.dimension
        }