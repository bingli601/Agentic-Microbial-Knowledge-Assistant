from Bio import Entrez
import json
import time
from tqdm import tqdm
from pathlib import Path

# ========== 配置区 ==========
Entrez.email = "bli@aggies.ncat.edu"      # 必须改
Entrez.api_key = "f05c76b7a6e08142fe409b92a9ecac641a08"         # 强烈建议填写

MAX_PER_QUERY = 120          # 每个关键词最多取多少
OUTPUT_FILE = Path("data/raw/pubmed_abstracts.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

queries = [
    "microbial survival modeling",
    "predictive microbiology food safety",
    "Listeria monocytogenes survival temperature",
    "Salmonella inactivation pH water activity",
    "bacterial survival modeling food",
    "D-value z-value microbial inactivation",
    "Listeria survival refrigerated food",
    "pathogen survival modeling low moisture",
    "microbial growth modeling food safety",
    "thermal inactivation kinetics bacteria"
]
# ===========================

def search_ids(query, max_results=120):
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance",
        retmode="xml"
    )
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def fetch_details(id_list):
    results = []
    batch_size = 50

    for i in tqdm(range(0, len(id_list), batch_size), desc="Fetching abstracts"):
        batch = id_list[i:i + batch_size]
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="abstract",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()
        except Exception as e:
            print(f"Batch error: {e}")
            time.sleep(2)
            continue

        for paper in records.get("PubmedArticle", []):
            try:
                medline = paper["MedlineCitation"]
                article = medline["Article"]

                # 摘要
                abstract = ""
                if "Abstract" in article and "AbstractText" in article["Abstract"]:
                    abstract_text = article["Abstract"]["AbstractText"]
                    if isinstance(abstract_text, list):
                        abstract = " ".join([str(t) for t in abstract_text])
                    else:
                        abstract = str(abstract_text)

                if len(abstract.strip()) < 150:  # 过滤太短的摘要
                    continue

                pmid = str(medline["PMID"])
                title = str(article.get("ArticleTitle", "")).strip()
                journal = ""
                year = ""

                if "Journal" in article:
                    journal = str(article["Journal"].get("Title", ""))
                    pub_date = article["Journal"].get("JournalIssue", {}).get("PubDate", {})
                    year = str(pub_date.get("Year", pub_date.get("MedlineDate", "")[:4]))

                results.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract.strip(),
                    "journal": journal,
                    "year": year,
                    "source": "pubmed"
                })
            except Exception:
                continue

        time.sleep(0.35)  # 礼貌限速

    return results

def main():
    all_ids = set()
    for q in queries:
        print(f"Searching: {q}")
        ids = search_ids(q, max_results=MAX_PER_QUERY)
        all_ids.update(ids)
        print(f"  → {len(ids)} ids")
        time.sleep(0.5)

    print(f"\nTotal unique PMIDs: {len(all_ids)}")
    data = fetch_details(list(all_ids))

    # 再次去重（按 pmid）
    unique = {}
    for item in data:
        unique[item["pmid"]] = item
    final_data = list(unique.values())

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n成功保存 {len(final_data)} 条摘要到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()