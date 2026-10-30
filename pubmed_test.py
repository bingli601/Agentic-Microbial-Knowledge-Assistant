import json

data = []
with open("data/raw/pubmed_abstracts.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

print(f"总条数: {len(data)}")
print(f"平均摘要长度: {sum(len(d['abstract']) for d in data) / len(data):.0f}")
print("\n随机看3条标题：")
for d in data[::len(data)//3][:3]:
    print("-", d["title"][:100])