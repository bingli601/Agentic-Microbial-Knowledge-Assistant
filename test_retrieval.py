from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ==================== 配置 ====================
COLLECTION_NAME = "microbial_rag"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QDRANT_PATH = "./qdrant_data"
TOP_K = 3
# ==============================================

def main():
    print("1. 加载 Embedding 模型...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    print("2. 连接本地向量库...")
    client = QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    retriever = index.as_retriever(similarity_top_k=TOP_K)

    # 测试问题（可以根据你的兴趣修改）
    test_questions = [
        "Listeria monocytogenes survival at refrigeration temperature",
        "Effect of pH on Salmonella inactivation",
        "What is D-value in predictive microbiology?",
        "Microbial survival modeling in low moisture foods",
        "Thermal inactivation kinetics of bacteria"
    ]

    for i, question in enumerate(test_questions, 1):
        print("\n" + "=" * 80)
        print(f"测试问题 {i}: {question}")
        print("=" * 80)

        nodes = retriever.retrieve(question)

        for rank, node in enumerate(nodes, 1):
            print(f"\n【第 {rank} 名】相似度: {node.score:.4f}")
            print(f"PMID   : {node.metadata.get('pmid', 'N/A')}")
            print(f"标题   : {node.metadata.get('title', 'N/A')}")
            print(f"年份   : {node.metadata.get('year', 'N/A')}")
            print(f"内容片段: {node.text[:300]}...")
            print("-" * 60)

if __name__ == "__main__":
    main()