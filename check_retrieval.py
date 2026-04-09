from src.pipeline import RAGPipeline

pipeline = RAGPipeline.from_config('config/settings.yaml')
chunks = pipeline.retriever.retrieve("在Python源文件中可以使用非ASCII编码吗", top_k=5)

print("检索到的内容：\n")
for i, c in enumerate(chunks):
    print(f"[{i+1}] score={c.final_score:.3f}")
    print(c.text[:300])
    print("---")
