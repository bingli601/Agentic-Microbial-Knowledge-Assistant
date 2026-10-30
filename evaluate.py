import json
from pathlib import Path
from datetime import datetime

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient

# ==================== 配置 ====================
COLLECTION_NAME = "microbial_rag"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QDRANT_PATH = "./qdrant_data"
LLM_MODEL = "qwen2.5:7b"
GOLDEN_FILE = Path("data/eval/golden_dataset.jsonl")
OUTPUT_FILE = Path("data/eval/evaluation_results.jsonl")
TOP_K = 4
# ==============================================

def build_query_engine():
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    llm = Ollama(model=LLM_MODEL, request_timeout=600.0, temperature=0.1)
    Settings.llm = llm

    client = QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

    qa_prompt = (
        "你是一个专业的微生物与食品安全助手。请主要依据下面提供的上下文回答问题。"
        "可以对信息进行归纳，但不要添加上下文中没有的具体数据。"
        "如果上下文没有相关信息，请说“根据现有文献，我无法确定”。"
        "用中文回答，并在最后列出参考的 PMID 和标题。\n\n"
        "上下文：\n{context_str}\n\n问题：{query_str}\n\n回答："
    )

    from llama_index.core import PromptTemplate
    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        text_qa_template=PromptTemplate(qa_prompt),
        response_mode="compact"
    )
    return query_engine

def main():
    print("加载评估数据...")
    samples = []
    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"共 {len(samples)} 个问题\n")

    print("初始化系统...")
    query_engine = build_query_engine()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for i, sample in enumerate(samples, 1):
        print("=" * 70)
        print(f"[{i}/{len(samples)}] {sample['id']}: {sample['question']}")
        print("-" * 70)

        response = query_engine.query(sample["question"])
        answer = response.response

        print("系统回答：")
        print(answer)
        print()

        # 显示参考答案，方便你对比打分
        print("参考答案：")
        print(sample["reference_answer"])
        print()

        # 人工打分
        while True:
            try:
                acc = int(input("准确性 (0-2): "))
                comp = int(input("完整性 (0-2): "))
                faith = int(input("忠实度/无幻觉 (0-2): "))
                if all(0 <= x <= 2 for x in [acc, comp, faith]):
                    break
                print("请输入 0、1 或 2")
            except ValueError:
                print("请输入数字 0、1 或 2")

        total = acc + comp + faith
        print(f"本题得分: {total}/6\n")

        record = {
            "id": sample["id"],
            "question": sample["question"],
            "question_type": sample.get("question_type"),
            "difficulty": sample.get("difficulty"),
            "reference_answer": sample["reference_answer"],
            "system_answer": answer,
            "score_accuracy": acc,
            "score_completeness": comp,
            "score_faithfulness": faith,
            "score_total": total,
            "timestamp": datetime.now().isoformat()
        }
        results.append(record)

        # 实时保存，防止中断丢失
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 汇总
    print("\n" + "=" * 70)
    print("评估完成！汇总结果：")
    avg = sum(r["score_total"] for r in results) / len(results)
    print(f"平均分: {avg:.2f} / 6")
    print(f"详细结果已保存到: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()