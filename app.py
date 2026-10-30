import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="微生物知识助手",
    page_icon="🦠",
    layout="centered"
)

st.title("🦠 微生物知识助手")
st.caption("基于 PubMed 文献的本地 RAG 问答系统 | Microbial Knowledge Assistant")

# ==================== 配置 ====================
COLLECTION_NAME = "microbial_rag"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QDRANT_PATH = "./qdrant_data"
LLM_MODEL = "qwen2.5:7b"
TOP_K = 4

# ==================== 加载系统（只加载一次） ====================
@st.cache_resource
def load_query_engine():
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    llm = Ollama(
        model=LLM_MODEL,
        request_timeout=600.0,
        temperature=0.1
    )
    Settings.llm = llm

    client = QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    qa_prompt = PromptTemplate(
        """你是一个专业的微生物与食品安全助手。请遵循以下规则：
    1. 主要依据下面提供的上下文回答问题。
    2. 可以归纳总结，但不要添加上下文中没有的具体数据。
    3. 如果上下文没有相关信息，请说“根据现有文献，我无法确定”。
    4. 用中文回答。
    5. 不要在回答中列出参考文献或 PMID，来源会由系统单独展示。

    上下文：
    {context_str}

    问题：{query_str}

    回答："""
    )

    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        text_qa_template=qa_prompt,
        response_mode="compact"
    )
    return query_engine

# ==================== 主界面 ====================
with st.spinner("正在加载模型与知识库，请稍候..."):
    try:
        query_engine = load_query_engine()
        st.success("系统已就绪")
    except Exception as e:
        st.error(f"加载失败: {e}")
        st.info("请确认：1) Ollama 已启动 2) qdrant_data 存在 3) 没有其他程序占用向量库")
        st.stop()

# 示例问题
example_questions = [
    "Listeria monocytogenes 在冷藏温度下能否生长？",
    "什么是 D-value？它在预测微生物学中有什么意义？",
    "低水分食品中 Salmonella 的存活主要受哪些因素影响？",
    "水分活度如何影响细菌的热耐受性？",
]

st.subheader("快速提问")
cols = st.columns(2)
for i, q in enumerate(example_questions):
    if cols[i % 2].button(q, use_container_width=True):
        st.session_state["question"] = q

# 输入框
question = st.text_input(
    "请输入你的问题：",
    value=st.session_state.get("question", ""),
    placeholder="例如：什么是预测微生物学？"
)

if st.button("提交", type="primary") or (question and st.session_state.get("auto_ask")):
    if not question.strip():
        st.warning("请输入问题")
    else:
        with st.spinner("正在检索文献并生成回答..."):
            try:
                response = query_engine.query(question.strip())
                st.markdown("### 回答")
                st.write(response.response)

                if response.source_nodes:
                    with st.expander("查看检索到的文献来源"):
                        for i, node in enumerate(response.source_nodes, 1):
                            st.markdown(
                                f"**{i}. PMID: {node.metadata.get('pmid')}**  \n"
                                f"{node.metadata.get('title', '')}"
                            )
            except Exception as e:
                st.error(f"生成失败: {e}")

st.divider()
st.caption("数据来源：PubMed 微生物相关论文摘要 | 模型：bge-small-en-v1.5 + qwen2.5:7b | 完全本地运行")