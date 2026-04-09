"""
FinetuneDataGenerator — 合成问答对生成模块

参考论文：Arzideh et al., JMIR 2026 (MIRACLE)
流程：
  1. 加载文档并切分（450字符/80重叠，与论文对齐）
  2. 对每个 chunk 调用 LLM 生成 5 个问答对
  3. 自动过滤（格式校验 + 幻觉检测）
  4. 保存为 JSONL 格式，供微调脚本使用
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from loguru import logger

from ..ingestion.loader import DataLoader
from ..ingestion.splitter import RecursiveCharacterSplitter
from ..generation.generator import OpenAIGenerator, GenerationConfig


# =========================================
# 问答对生成 Prompt
# =========================================
QA_GENERATION_PROMPT = """你是一个专业的文档分析专家。请基于以下文档片段，生成5个相关问题及对应答案。

要求：
1. 每个问题必须以问号结尾
2. 答案必须能从文档片段中直接找到依据，不得编造
3. 问题应覆盖文档中的关键信息（专有名词、参数、操作规程、定义等）
4. 避免询问行政信息（如姓名、日期、编号等无实质内容的问题）
5. 严格按照以下格式输出，不要有其他内容

文档片段：
{chunk}

请按以下格式输出：
1. 问题：[问题内容]
   答案：[答案内容]
2. 问题：[问题内容]
   答案：[答案内容]
3. 问题：[问题内容]
   答案：[答案内容]
4. 问题：[问题内容]
   答案：[答案内容]
5. 问题：[问题内容]
   答案：[答案内容]"""

# 过滤掉的行政类关键词
ADMIN_KEYWORDS = ["姓名", "出生日期", "身份证", "联系方式", "地址", "电话", "邮箱"]


class FinetuneDataGenerator:
    """
    合成问答对生成器

    用法：
        generator = FinetuneDataGenerator.from_config("config/settings.yaml")
        generator.generate("data/raw/", output_path="data/finetune/qa_pairs.jsonl")
    """

    def __init__(self, llm_generator, chunk_size: int = 450, chunk_overlap: int = 80):
        self.llm = llm_generator
        self.splitter = RecursiveCharacterSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=50
        )
        self.data_loader = DataLoader()

    @classmethod
    def from_config(cls, config_path: str) -> "FinetuneDataGenerator":
        import os
        import yaml
        from dotenv import load_dotenv
        load_dotenv()
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        def expand_env_vars(obj):
            if isinstance(obj, dict):
                return {k: expand_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [expand_env_vars(i) for i in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                env_var = obj[2:-1]
                default = None
                if ':' in env_var:
                    env_var, default = env_var.split(':', 1)
                return os.getenv(env_var, default)
            return obj

        config = expand_env_vars(config)
        llm_cfg = config.get("llm", {})
        llm = OpenAIGenerator(
            model=llm_cfg.get("model", "deepseek-chat"),
            api_key=llm_cfg.get("api_key"),
            base_url=llm_cfg.get("base_url"),
        )
        return cls(llm)

    # --------------------------------------------------
    # 主流程
    # --------------------------------------------------
    def generate(
        self,
        source: str,
        output_path: str = "data/finetune/qa_pairs.jsonl",
        max_chunks: Optional[int] = None
    ) -> int:
        """
        从文档目录生成问答对并保存。

        Args:
            source: 文档目录或单个文件路径
            output_path: 输出 JSONL 文件路径
            max_chunks: 最多处理的 chunk 数（None 表示全部，调试时可设小值）

        Returns:
            保存的有效问答对数量
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 加载并切分文档
        path = Path(source)
        if path.is_dir():
            documents = self.data_loader.load_directory(path)
        else:
            documents = [self.data_loader.load_document(path)]

        all_chunks = self.splitter.split_batch(documents)
        logger.info(f"Total chunks to process: {len(all_chunks)}")

        if max_chunks:
            all_chunks = all_chunks[:max_chunks]
            logger.info(f"Limited to {max_chunks} chunks for this run")

        total_generated = 0
        total_filtered = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(all_chunks):
                if not chunk.content.strip() or len(chunk.content) < 50:
                    continue

                try:
                    qa_pairs = self._generate_qa_for_chunk(chunk.content)
                    valid_pairs = self._filter_qa_pairs(qa_pairs, chunk.content)

                    for question, answer in valid_pairs:
                        record = {
                            "question": question,
                            "chunk": chunk.content,
                            "answer": answer,
                            "metadata": {
                                "source": chunk.metadata.get("filename", ""),
                                "page": chunk.metadata.get("page", -1),
                                "section": chunk.metadata.get("section_path", ""),
                            }
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_generated += 1

                    filtered = 5 - len(valid_pairs)
                    total_filtered += filtered

                    if (i + 1) % 50 == 0:
                        logger.info(
                            f"Progress: {i+1}/{len(all_chunks)} chunks, "
                            f"{total_generated} valid pairs, "
                            f"{total_filtered} filtered"
                        )

                except Exception as e:
                    logger.warning(f"Chunk {i} failed: {e}")
                    continue

        logger.info(
            f"Done. Generated {total_generated} valid pairs, "
            f"filtered {total_filtered} "
            f"({total_filtered/(total_generated+total_filtered)*100:.1f}%)"
        )
        return total_generated

    # --------------------------------------------------
    # 单个 chunk 生成问答对
    # --------------------------------------------------
    def _generate_qa_for_chunk(self, chunk: str) -> List[Tuple[str, str]]:
        prompt = QA_GENERATION_PROMPT.format(chunk=chunk)
        config = GenerationConfig(temperature=0.3, max_tokens=800)
        result = self.llm.generate(prompt, config)
        return self._parse_qa_output(result.text)

    # --------------------------------------------------
    # 解析 LLM 输出
    # --------------------------------------------------
    @staticmethod
    def _parse_qa_output(text: str) -> List[Tuple[str, str]]:
        """
        解析 LLM 输出的问答对格式：
        1. 问题：...
           答案：...
        """
        pairs = []
        # 匹配 "数字. 问题：... 答案：..." 的模式
        pattern = re.compile(
            r'\d+\.\s*问题[：:]\s*(.+?)\s*答案[：:]\s*(.+?)(?=\d+\.\s*问题|$)',
            re.DOTALL
        )
        for m in pattern.finditer(text):
            question = m.group(1).strip()
            answer = m.group(2).strip()
            if question and answer:
                pairs.append((question, answer))
        return pairs

    # --------------------------------------------------
    # 过滤问答对
    # --------------------------------------------------
    @staticmethod
    def _filter_qa_pairs(
        pairs: List[Tuple[str, str]],
        chunk: str
    ) -> List[Tuple[str, str]]:
        """
        过滤规则（参考论文）：
        1. 问题必须以问号结尾
        2. 答案不能为空
        3. 答案关键词必须能在 chunk 中找到（防幻觉）
        4. 过滤行政类问题
        """
        valid = []
        for question, answer in pairs:
            # 规则1：问题以问号结尾
            if not (question.endswith("？") or question.endswith("?")):
                continue
            # 规则2：答案不为空
            if len(answer.strip()) < 5:
                continue
            # 规则3：答案关键词能在 chunk 中找到（取答案前20字的关键词）
            answer_keywords = [w for w in answer[:20] if '\u4e00' <= w <= '\u9fff']
            if answer_keywords:
                found = sum(1 for kw in answer_keywords if kw in chunk)
                if found / len(answer_keywords) < 0.3:
                    continue  # 答案与 chunk 重叠率过低，疑似幻觉
            # 规则4：过滤行政类问题
            if any(kw in question for kw in ADMIN_KEYWORDS):
                continue
            valid.append((question, answer))
        return valid


# =========================================
# 数据集加载工具（供微调脚本使用）
# =========================================
def load_qa_dataset(jsonl_path: str, train_ratio: float = 0.8) -> Tuple[List, List]:
    """
    加载 JSONL 问答对，按文档来源划分训练/测试集（避免数据泄露）。
    返回 (train_pairs, test_pairs)，每项为 (question, chunk) 元组。
    """
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    # 按来源文件分组，整文件划分，避免同文件数据同时出现在训练和测试集
    from collections import defaultdict
    by_source: Dict[str, List] = defaultdict(list)
    for r in records:
        source = r["metadata"].get("source", "unknown")
        by_source[source].append((r["question"], r["chunk"]))

    sources = list(by_source.keys())
    split_idx = int(len(sources) * train_ratio)
    train_sources = sources[:split_idx]
    test_sources = sources[split_idx:]

    train_pairs = [pair for s in train_sources for pair in by_source[s]]
    test_pairs = [pair for s in test_sources for pair in by_source[s]]

    logger.info(
        f"Dataset loaded: {len(train_pairs)} train pairs, "
        f"{len(test_pairs)} test pairs"
    )
    return train_pairs, test_pairs
