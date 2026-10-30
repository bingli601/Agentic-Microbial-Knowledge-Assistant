import json
from pathlib import Path
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ==================== 配置 ====================
PROCESSED_FILE = Path("data/processed/docs.jsonl")
COLLECTION_NAME = "microbial_rag"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
QDRANT_PATH = "./qdrant_data"          # 本地存储路径
# ==============================================

def load_documents():
    docs = []
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            doc = Document(
                text=item["text"],
                metadata={
                    "pmid": item["pmid"],
                    "title": item["title"],
                    "journal": item.get("journal", ""),
                    "year": item.get("year", ""),
                    "source": item.get("source", "pubmed")
                },
                id_=item["pmid"]
            )
            docs.append(doc)
    return docs

def main():
    print("1. 加载文档...")
    documents = load_documents()
    print(f"   共加载 {len(documents)} 篇文档")

    print("2. 初始化本地 Embedding 模型...")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        trust_remote_code=True
    )
    Settings.embed_model = embed_model

    print("3. 初始化本地 Qdrant（无需 Docker）...")
    # 使用本地路径模式，数据会保存在 ./qdrant_data 文件夹
    client = QdrantClient(path=QDRANT_PATH)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("4. 分块 + 建立索引（本地计算）...")
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[node_parser],
        show_progress=True
    )

    print("\n索引构建完成！")
    print(f"集合名称: {COLLECTION_NAME}")
    print(f"数据存储位置: {QDRANT_PATH}")
    print(f"使用模型: {EMBED_MODEL_NAME}")

if __name__ == "__main__":
    main()