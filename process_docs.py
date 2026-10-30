import json
from pathlib import Path
from tqdm import tqdm

RAW_FILE = Path("data/raw/pubmed_abstracts.jsonl")
PROCESSED_FILE = Path("data/processed/docs.jsonl")
PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    if not text:
        return ""
    # 基础清洗
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())  # 去掉多余空格
    return text.strip()

def main():
    raw_data = []
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(json.loads(line))

    print(f"原始数据: {len(raw_data)} 条")

    cleaned = []
    seen_pmids = set()

    for item in tqdm(raw_data, desc="Cleaning"):
        pmid = str(item.get("pmid", "")).strip()
        if not pmid or pmid in seen_pmids:
            continue

        title = clean_text(item.get("title", ""))
        abstract = clean_text(item.get("abstract", ""))

        # 过滤条件
        if len(abstract) < 200:          # 摘要太短
            continue
        if len(title) < 15:              # 标题异常
            continue

        # 组合最终用于检索的文本
        text = f"{title}. {abstract}"

        cleaned.append({
            "id": pmid,
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "text": text,
            "journal": item.get("journal", ""),
            "year": item.get("year", ""),
            "source": "pubmed"
        })
        seen_pmids.add(pmid)

    # 保存
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        for doc in cleaned:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\n清洗完成！")
    print(f"有效数据: {len(cleaned)} 条")
    print(f"已保存到: {PROCESSED_FILE}")

    # 简单统计
    avg_len = sum(len(d["text"]) for d in cleaned) / len(cleaned)
    print(f"平均 text 长度: {avg_len:.0f} 字符")

if __name__ == "__main__":
    main()