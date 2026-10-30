import json
from pathlib import Path
from collections import defaultdict

from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
import numpy as np

# ==================== 配置 ====================
COLLECTION_NAME = "microbial_rag"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QDRANT_PATH = "./qdrant_data"
PROCESSED_FILE = Path("data/processed/docs.jsonl")
LLM_MODEL = "qwen2.5:7b"
TOP_K = 5
# ==============================================

class HybridRetriever:
    def __init__(self, vector_retriever, nodes, top_k=5):
        self.vector_retriever = vector_retriever
        self.top_k = top_k
        self.nodes = nodes
        self.corpus = [node.text.lower().split() for node in nodes]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str):
        # 向量检索
        vector_nodes = self.vector_retriever.retrieve(query)

        # BM25 检索
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:self.top_k]

        bm25_nodes = []
        for idx in top_bm25_idx:
            if bm25_scores[idx] > 0:
                node = self.nodes[idx]
                bm25_nodes.append(NodeWithScore(node=node, score=float(bm25_scores[idx])))

        # RRF 融合
        scores = defaultdict(float)
        node_map = {}

        for rank, n in enumerate(vector_nodes):
            node_id = n.node.node_id
            scores[node_id] += 1.0 / (60 + rank)
            node_map[node_id] = n.node

        for rank, n in enumerate(bm25_nodes):
            node_id = n.node.node_id
            scores[node_id] += 1.0 / (60 + rank)
            node_map[node_id] = n.node

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.top_k]
        return [NodeWithScore(node=node_map[nid], score=scores[nid]) for nid in sorted_ids]

def load_all_nodes():
    nodes = []
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            node = TextNode(
                text=item["text"],
                id_=item["pmid"],
                metadata={
                    "pmid": item["pmid"],
                    "title": item["title"],
                    "journal": item.get("journal", ""),
                    "year": item.get("year", "")
                }
            )
            nodes.append(node)
    return nodes

def rewrite_query(llm, question: str) -> list:
    """把用户问题改写成多个更适合检索的英文查询"""
    prompt = f"""你是一个检索专家。请把下面的问题改写成 3 个更适合在科学文献中检索的英文查询。
要求：
1. 每个查询一行
2. 使用专业术语
3. 如果是定义类问题，请包含 definition、what is、meaning 等词
4. 只输出查询，不要编号，不要解释

问题：{question}
"""
    response = llm.complete(prompt)
    lines = [line.strip() for line in response.text.strip().split("\n") if line.strip()]
    # 过滤掉可能的编号
    queries = []
    for line in lines:
        clean = line.lstrip("0123456789.-、）) ").strip()
        if clean:
            queries.append(clean)
    return queries[:3] if queries else [question]

def merge_nodes(list_of_node_lists, top_k=6):
    """合并多次检索结果并去重"""
    scores = defaultdict(float)
    node_map = {}

    for nodes in list_of_node_lists:
        for rank, n in enumerate(nodes):
            node_id = n.node.node_id
            scores[node_id] += 1.0 / (60 + rank)
            node_map[node_id] = n.node

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
    return [NodeWithScore(node=node_map[nid], score=scores[nid]) for nid in sorted_ids]

def format_context(nodes):
    parts = []
    for i, node in enumerate(nodes, 1):
        pmid = node.metadata.get("pmid", "N/A")
        title = node.metadata.get("title", "N/A")
        parts.append(f"[{i}] PMID: {pmid} | {title}\n{node.text[:900]}")
    return "\n\n".join(parts)

def build_engine():
    print("加载 Embedding 模型...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    print("加载本地大模型...")
    llm = Ollama(model=LLM_MODEL, request_timeout=600.0, temperature=0.1)
    Settings.llm = llm

    print("连接向量库...")
    client = QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    vector_retriever = index.as_retriever(similarity_top_k=TOP_K)

    print("构建 Hybrid Retriever...")
    nodes = load_all_nodes()
    hybrid_retriever = HybridRetriever(vector_retriever, nodes, top_k=TOP_K)

    return hybrid_retriever, llm

def main():
    hybrid_retriever, llm = build_engine()

    qa_prompt_template = """你是一个专业的微生物与食品安全助手。请遵循以下规则回答：

1. 主要依据下面提供的【上下文】回答问题。
2. 可以对上下文中的信息进行归纳和总结，但不要添加上下文中没有的具体数据或结论。
3. 如果上下文确实没有相关信息，请明确说“根据现有文献，我无法确定”。
4. 使用中文回答。
5. 在回答最后列出实际参考的文献，格式必须为：
   - PMID: 数字 | 文献标题
6. 只能使用上下文中出现的 PMID 和标题，禁止编造。

【上下文】
{context}

【问题】
{question}

【回答】
"""

    print("\n" + "="*60)
    print("Hybrid + Query Rewrite RAG 系统已启动（输入 exit 退出）")
    print("="*60)

    while True:
        question = input("\n请输入你的问题: ").strip()
        if question.lower() in ["exit", "quit", "退出"]:
            break
        if not question:
            continue

        print("\n1. 正在改写查询...")
        rewritten_queries = rewrite_query(llm, question)
        print("   改写结果:")
        for i, q in enumerate(rewritten_queries, 1):
            print(f"   {i}. {q}")

        print("\n2. 正在进行 Hybrid 检索...")
        all_results = []
        for q in rewritten_queries:
            nodes = hybrid_retriever.retrieve(q)
            all_results.append(nodes)

        # 也用原始问题检索一次
        all_results.append(hybrid_retriever.retrieve(question))

        final_nodes = merge_nodes(all_results, top_k=6)
        context = format_context(final_nodes)

        print("3. 正在生成回答...\n")
        prompt = qa_prompt_template.format(context=context, question=question)
        response = llm.complete(prompt)

        print("-" * 60)
        print("回答：")
        print(response.text)
        print("-" * 60)

        print("\n最终检索来源：")
        for i, node in enumerate(final_nodes, 1):
            print(f"{i}. PMID: {node.metadata.get('pmid')} | {node.metadata.get('title')[:80]}...")

if __name__ == "__main__":
    main()