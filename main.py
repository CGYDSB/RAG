#!/usr/bin/env python3
"""
main.py 是 RAG 系统的本地命令行入口（CLI Layer），
通过 argparse 将系统能力封装为四个核心命令：文档索引、问答查询、API 服务启动以及模型评估。
它直接调用 RAGPipeline 完成检索与生成流程，不依赖 FastAPI，使得同一套核心 RAG 能力既可以用于生产 API 服务，
也可以用于本地开发调试与批量评测，是连接开发与运行阶段的统一控制入口。

RAG System - Local Runner（本地运行入口）

这个文件是整个 RAG 系统的“命令行控制中心（CLI）”

支持 4 大能力：
1. index     → 文档索引
2. query     → 本地问答
3. server    → 启动 API 服务
4. evaluate  → 模型评估

特点：
- 不依赖 FastAPI
- 直接调用 RAGPipeline
- 用于开发 / 调试 / 本地运行
"""

import argparse
import sys
import os

# =========================
# 1. 添加 src 路径（确保能 import 本地模块）
# =========================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
import yaml
from src.pipeline import RAGPipeline


# =========================
# 2. 日志配置
# =========================
def setup_logging():
    """
    配置日志系统（Loguru）

    输出：
    - 控制台日志（INFO）
    - 文件日志（DEBUG）
    """
    logger.remove()

    # 控制台日志
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level="INFO"
    )

    # 文件日志（用于排查问题）
    logger.add(
        "logs/rag_system.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG"
    )


# =========================
# 3. 配置加载
# =========================
def load_config():
    """
    加载配置文件 config/settings.yaml

    如果不存在 → 直接退出（CLI 不允许 fallback）
    """
    config_path = os.path.join('config', 'settings.yaml')

    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =========================
# 4. index 命令（文档入库）
# =========================
def index_command(args):
    """
    功能：把文档写入向量数据库

    流程：
    1. 读取数据源
    2. 初始化 RAGPipeline
    3. （可选）清空向量库
    4. 文档切分 + embedding + 存储
    """
    logger.info(f"Indexing documents from: {args.source}")

    config = load_config()
    pipeline = RAGPipeline.from_config('config/settings.yaml')

    if args.clear:
        pipeline.vector_store.clear()
        logger.info("Vector store cleared.")

    pipeline.index_documents(
        args.source,
        batch_size=args.batch_size
    )

    stats = pipeline.get_stats()
    logger.info(f"Indexing complete. Total chunks: {stats['vector_count']}")


# =========================
# 5. query 命令（本地问答）
# =========================
def query_command(args):
    """
    功能：本地 RAG 问答测试

    支持：
    - 普通输出
    - 流式输出
    """
    config = load_config()
    pipeline = RAGPipeline.from_config('config/settings.yaml')

    logger.info(f"Query: {args.question}")

    response = pipeline.query(
        question=args.question,
        top_k=args.top_k,
        stream=args.stream
    )

    # =========================
    # 5.1 流式输出模式
    # =========================
    if args.stream:
        print("\nAnswer: ", end="", flush=True)

        for chunk in response:
            if chunk.startswith("__SOURCES__:"):
                continue
            print(chunk, end="", flush=True)

        print()

    # =========================
    # 5.2 普通模式
    # =========================
    else:
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(response.answer)

        print("\n" + "=" * 60)
        print("SOURCES:")
        print("=" * 60)

        for source in response.sources:
            print(f"[{source['index']}] Score: {source.get('score', source.get('relevance_score', 0)):.3f}")
            print(f"    Source: {source['metadata'].get('source', 'Unknown')}")
            print(f"    Page:   {source['metadata'].get('page', 'N/A')}")

            if args.verbose:
                print(f"    Text: {source.get('text', 'N/A')[:200]}...")

            print()


# =========================
# 6. chat 命令（多轮对话）
# =========================
def chat_command(args):
    """
    功能：交互式多轮对话
    - 自动携带对话历史
    - 超出窗口时 LLM 压缩历史
    - 输入 /clear 清空历史，/history 查看历史，/quit 退出
    """
    pipeline = RAGPipeline.from_config('config/settings.yaml')

    print("\n" + "=" * 60)
    print("RAG 多轮对话模式")
    print("=" * 60)
    print("指令：/clear 清空历史  /history 查看历史  /quit 退出")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("你：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出对话。")
            break

        if not question:
            continue

        # ===== 内置指令 =====
        if question == "/quit":
            print("已退出对话。")
            break

        if question == "/clear":
            pipeline.conversation.clear()
            print("[系统] 对话历史已清空。\n")
            continue

        if question == "/history":
            history = pipeline.conversation.history
            if not history:
                print("[系统] 暂无对话历史。\n")
            else:
                print(f"\n[对话历史] 共 {len(history)} 轮")
                if pipeline.conversation.summary:
                    print(f"[早期摘要] {pipeline.conversation.summary}")
                for t in pipeline.conversation.get_recent_turns():
                    print(f"  第{t.turn_id}轮 Q: {t.question}")
                    print(f"  第{t.turn_id}轮 A: {t.answer[:100]}...")
                print()
            continue

        # ===== 正常问答 =====
        turn_num = len(pipeline.conversation.history) + 1
        logger.info(f"[第{turn_num}轮] {question}")

        print(f"\n助手：", end="", flush=True)

        if getattr(args, 'stream', False):
            full_answer = ""
            sources = []
            for token in pipeline.query(question=question, top_k=args.top_k, stream=True, use_history=True):
                if token.startswith("__SOURCES__:"):
                    import json as _json
                    sources = _json.loads(token[len("__SOURCES__:"):].strip())
                else:
                    print(token, end="", flush=True)
                    full_answer += token
            print()
            pipeline.conversation.add_turn(question, full_answer)
        else:
            response = pipeline.query(question=question, top_k=args.top_k, use_history=True)
            print(response.answer)
            sources = response.sources

            if getattr(args, 'verbose', False) and sources:
                print("\n  [检索原文]")
                for s in sources:
                    filename = s['metadata'].get('filename', '未知文件')
                    page = s['metadata'].get('page', '?')
                    print(f"    [{s['index']}] {filename} p.{page}: {s.get('text', '')[:150]}...")

        # 拒答时不显示来源（避免误导用户）
        is_off_topic = "超出了我的知识范围" in (response.answer if not getattr(args, 'stream', False) else "")
        if getattr(args, 'show_sources', False) and sources and not is_off_topic:
            print("\n  来源：")
            for s in sources:
                filename = s['metadata'].get('filename', s['metadata'].get('source', '未知文件'))
                page = s['metadata'].get('page', '?')
                section = s.get('section_path', '') or s['metadata'].get('section_path', '')
                score = s.get('score', 0)
                location = f"第{page}页"
                if section:
                    location = f"{section} · 第{page}页"
                print(f"    [[{s['index']}]] {filename} | {location} | 相关度 {score:.3f}")

        # 显示历史压缩提示
        if len(pipeline.conversation.history) > pipeline.conversation.keep_recent:
            if pipeline.conversation.summary:
                print(f"  [历史已压缩，摘要长度: {len(pipeline.conversation.summary)}字]")

        print()


# =========================
# 7. server 命令（启动 API）
# =========================
def server_command(args):
    """
    启动 FastAPI 服务（uvicorn）
    """
    import uvicorn

    logger.info(f"Starting API server on {args.host}:{args.port}")

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )


# =========================
# 7. evaluate 命令（评估）
# =========================
def evaluate_command(args):
    """
    功能：批量评估 RAG 效果

    输入：
    - test_file: JSON 测试集

    输出：
    - faithfulness
    - relevancy
    - precision
    """
    logger.info(f"Running evaluation on: {args.test_file}")

    import json
    with open(args.test_file, 'r') as f:
        test_cases = json.load(f)

    config = load_config()
    pipeline = RAGPipeline.from_config('config/settings.yaml')

    results = []

    for case in test_cases:
        response = pipeline.query(case['question'], top_k=5)

        result = pipeline.evaluator.evaluate(
            question=case['question'],
            answer=response.answer,
            contexts=[s['text'] for s in response.sources],
            ground_truth=case.get('ground_truth')
        )

        results.append(result)

    # 生成评估报告
    report = pipeline.evaluator.generate_report(args.output)
    print(report)


# =========================
# 新增：generate-finetune-data 命令
# =========================
def generate_finetune_data_command(args):
    """
    功能：从文档库生成微调用的合成问答对

    流程：
    1. 加载文档并切分（450字符/80重叠）
    2. 对每个 chunk 调用 LLM 生成 5 个问答对
    3. 自动过滤（格式校验 + 幻觉检测）
    4. 保存为 JSONL 格式
    """
    from src.embedding.data_gen import FinetuneDataGenerator

    logger.info(f"Generating finetune data from: {args.source}")

    generator = FinetuneDataGenerator.from_config('config/settings.yaml')
    count = generator.generate(
        source=args.source,
        output_path=args.output,
        max_chunks=args.max_chunks
    )
    logger.info(f"Saved {count} valid QA pairs to: {args.output}")


# =========================
# 新增：finetune 命令
# =========================
def finetune_command(args):
    """
    功能：微调 Embedding 模型

    流程：
    1. 加载 JSONL 问答对数据集
    2. 按文档来源划分训练/测试集
    3. 使用 MultipleNegativesRankingLoss 微调
    4. 评估并对比微调前后性能
    """
    from src.embedding.data_gen import load_qa_dataset
    from src.embedding.finetuner import EmbeddingFinetuner

    logger.info(f"Loading dataset from: {args.data}")

    train_pairs, test_pairs = load_qa_dataset(args.data, train_ratio=0.8)

    finetuner = EmbeddingFinetuner(
        base_model=args.base_model,
        output_path=args.output,
        device=args.device
    )

    finetuner.train(
        train_pairs=train_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    if test_pairs:
        logger.info("Comparing fine-tuned model with baseline...")
        finetuner.compare_with_baseline(test_pairs, top_k=10)


# =========================
# 8. CLI 主入口
# =========================
def main():
    """
    CLI 入口函数

    整个程序的启动点，负责：
    1. 初始化日志系统
    2. 构建命令行参数解析器
    3. 注册所有子命令及其参数
    4. 解析用户输入并分发到对应的处理函数
    """
    # 初始化日志（控制台 + 文件双输出）
    setup_logging()

    # 创建顶层解析器
    # description：显示在 --help 的说明文字
    # RawDescriptionHelpFormatter：保留说明文字中的换行格式
    parser = argparse.ArgumentParser(
        description="RAG System - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 创建子命令容器
    # dest='command'：解析后可通过 args.command 获取用户输入的子命令名
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ===== index 子命令：文档入库 =====
    # 用法：python main.py index --source data/raw/
    index_parser = subparsers.add_parser('index', help='索引文档到向量数据库')
    index_parser.add_argument('--source', '-s', required=True,
                              help='文档路径（文件或目录）')
    index_parser.add_argument('--batch-size', '-b', type=int, default=100,
                              help='每批处理的 chunk 数量，默认100')
    index_parser.add_argument('--clear', action='store_true',
                              help='索引前清空向量库（删除所有旧数据）')
    index_parser.set_defaults(func=index_command)

    # ===== query 子命令：单次问答 =====
    # 用法：python main.py query "你的问题" [--top-k 5] [--stream] [--verbose]
    query_parser = subparsers.add_parser('query', help='单次 RAG 问答')
    query_parser.add_argument('question',
                              help='用户问题（直接跟在 query 后面）')  # 位置参数，不需要 --
    query_parser.add_argument('--top-k', '-k', type=int, default=5,
                              help='检索返回的文档数量，默认5')         # 影响召回率和上下文长度
    query_parser.add_argument('--stream', action='store_true',
                              help='开启流式输出（逐字打印）')          # 类似 ChatGPT 打字效果
    query_parser.add_argument('--verbose', '-v', action='store_true',
                              help='显示检索到的原文片段')              # 调试时查看检索内容
    query_parser.set_defaults(func=query_command)

    # ===== chat 子命令：多轮对话 =====
    # 用法：python main.py chat [--top-k 5] [--show-sources]
    # 进入交互模式后支持内置指令：/clear /history /quit
    chat_parser = subparsers.add_parser('chat', help='多轮交互对话（自动管理历史）')
    chat_parser.add_argument('--top-k', '-k', type=int, default=5,
                             help='每轮检索返回的文档数量，默认5')      # 多轮对话中每轮独立检索
    chat_parser.add_argument('--show-sources', '-s', action='store_true',
                             help='每轮回答后显示来源文件和页码')        # 方便追溯引用来源
    chat_parser.set_defaults(func=chat_command)

    # ===== server 子命令：启动 API 服务 =====
    # 用法：python main.py server [--host 0.0.0.0] [--port 8000] [--reload]
    server_parser = subparsers.add_parser('server', help='启动 FastAPI HTTP 服务')
    server_parser.add_argument('--host', default='0.0.0.0',
                               help='监听地址，默认 0.0.0.0（对外开放）')  # 改为 127.0.0.1 则只允许本机访问
    server_parser.add_argument('--port', '-p', type=int, default=8000,
                               help='监听端口，默认 8000')
    server_parser.add_argument('--reload', action='store_true',
                               help='开启热重载（修改代码自动重启，仅开发用）')  # 生产环境不要开启
    server_parser.add_argument('--workers', '-w', type=int, default=1,
                               help='并发 worker 数量，默认1')              # CPU 核数 × 2 + 1 是推荐值
    server_parser.set_defaults(func=server_command)

    # ===== evaluate 子命令：批量评估 =====
    # 用法：python main.py evaluate --test-file data/test_cases.json [--output report.txt]
    eval_parser = subparsers.add_parser('evaluate', help='批量评估 RAG 回答质量')
    eval_parser.add_argument('--test-file', '-t', required=True,
                             help='测试集 JSON 文件路径（包含 question/ground_truth 字段）')  # 必填
    eval_parser.add_argument('--output', '-o',
                             help='评估报告输出路径（不填则只打印到终端）')  # 可选，不填则只打印
    eval_parser.set_defaults(func=evaluate_command)

    # ===== generate-finetune-data 子命令：生成微调数据 =====
    # 用法：python main.py generate-finetune-data --source data/raw/ [--output data/finetune/qa_pairs.jsonl]
    gen_ft_parser = subparsers.add_parser(
        'generate-finetune-data',
        help='从文档库生成 Embedding 微调用的合成问答对（参考 MIRACLE 论文）'
    )
    gen_ft_parser.add_argument('--source', '-s', required=True,
                               help='文档路径（文件或目录）')
    gen_ft_parser.add_argument('--output', '-o',
                               default='data/finetune/qa_pairs.jsonl',
                               help='输出 JSONL 文件路径，默认 data/finetune/qa_pairs.jsonl')
    gen_ft_parser.add_argument('--max-chunks', type=int, default=None,
                               help='最多处理的 chunk 数（调试时可设小值，如 20）')
    gen_ft_parser.set_defaults(func=generate_finetune_data_command)

    # ===== finetune 子命令：微调 Embedding 模型 =====
    # 用法：python main.py finetune --data data/finetune/qa_pairs.jsonl
    ft_parser = subparsers.add_parser(
        'finetune',
        help='使用合成问答对微调 Embedding 模型（MultipleNegativesRankingLoss）'
    )
    ft_parser.add_argument('--data', '-d', required=True,
                           help='JSONL 问答对文件路径（由 generate-finetune-data 生成）')
    ft_parser.add_argument('--base-model', default='BAAI/bge-small-zh-v1.5',
                           help='基础模型名称或路径，默认 BAAI/bge-small-zh-v1.5')
    ft_parser.add_argument('--output', '-o', default='models/medical-bge-finetuned',
                           help='微调后模型保存路径，默认 models/medical-bge-finetuned')
    ft_parser.add_argument('--epochs', type=int, default=1,
                           help='训练轮数，默认 1（论文建议单 epoch 防过拟合）')
    ft_parser.add_argument('--batch-size', '-b', type=int, default=32,
                           help='批大小，CPU 建议 32-64，默认 32')
    ft_parser.add_argument('--lr', type=float, default=2e-5,
                           help='学习率，默认 2e-5（与论文一致）')
    ft_parser.add_argument('--device', default='cpu',
                           help='训练设备，cpu 或 cuda，默认 cpu')
    ft_parser.set_defaults(func=finetune_command)

    # 解析命令行参数
    args = parser.parse_args()

    # 如果用户没有输入任何子命令，打印帮助信息并退出
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 调用对应子命令的处理函数（由 set_defaults(func=...) 绑定）
    args.func(args)


# =========================
# 9. 程序入口
# =========================
if __name__ == "__main__":
    main()