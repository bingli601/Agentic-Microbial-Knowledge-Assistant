from llama_index.core import VectorStoreIndex, Settings, StorageContext, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient

# ==================== 配置 ====================
COLLECTION_NAME = "microbial_rag"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QDRANT_PATH = "./qdrant_data"
LLM_MODEL = "qwen2.5:7b"
TOP_K = 4
# ==============================================

def build_qa_engine():
    print("加载 Embedding 模型...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    print("加载本地大模型 (Ollama)...")
    llm = Ollama(
        model=LLM_MODEL,
        request_timeout=600.0,
        temperature=0.1
    )
    Settings.llm = llm

    print("连接向量库...")
    client = QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    # ==================== 优化后的提示词 ====================
    qa_prompt = PromptTemplate(
        """你是一个专业的微生物与食品安全助手。请遵循以下规则回答：

    1. 主要依据下面提供的【上下文】回答问题。
    2. 可以对上下文中的信息进行归纳和总结，但不要添加上下文中没有的具体数据或结论。
    3. 如果上下文确实没有相关信息，请明确说“根据现有文献，我无法确定”。
    4. 使用中文回答。
    5. 在回答最后列出实际参考的文献，格式必须为：
       - PMID: 数字 | 文献标题
    6. 只能使用上下文中出现的 PMID 和标题，禁止编造。

    【上下文】
    {context_str}

    【问题】
    {query_str}

    【回答】
    """
    )
    # ======================================================

    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        text_qa_template=qa_prompt,
        response_mode="compact"
    )

    return query_engine

def main():
    query_engine = build_qa_engine()

    print("\n" + "="*60)
    print("微生物知识问答系统已启动（输入 exit 退出）")
    print("="*60)

    while True:
        question = input("\n请输入你的问题: ").strip()
        if question.lower() in ["exit", "quit", "退出"]:
            print("已退出。")
            break
        if not question:
            continue

        print("\n正在检索并生成回答，请稍候...\n")
        response = query_engine.query(question)

        print("-" * 60)
        print("回答：")
        print(response.response)
        print("-" * 60)

        if response.source_nodes:
            print("\n系统检索到的原始来源：")
            for i, node in enumerate(response.source_nodes, 1):
                print(f"{i}. PMID: {node.metadata.get('pmid')} | {node.metadata.get('title')[:90]}...")

if __name__ == "__main__":
    main()