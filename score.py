# score.py — 직접 짠 채점기 (부분 문자열 일치 쓰지 않음). CPU만 씀.
import json, re, math
from gen_data import TOK, bm25_index

TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
ANS = re.compile(r"^\s*ANSWER:\s*(.+?)(?:\s+CITE:\s*(d\d+))?\s*$", re.S)
ABS = re.compile(r"^\s*ABSTAIN:\s*insufficient_evidence\s*$")

def norm(s): return " ".join(TOK(s or ""))

def parse1(out):
    m = TC.search(out)
    if m:
        try:
            j = json.loads(m.group(1)); return "search", j.get("arguments", {}).get("query", "")
        except Exception: return "invalid", None
    m = ANS.match(out.strip().split("\n")[0])
    return ("direct", m.group(1)) if m else ("invalid", None)

def parse2(out):
    line = out.strip().split("\n")[0]
    if ABS.match(line): return "abstain", None, None
    m = ANS.match(line)
    return ("answer", m.group(1), m.group(2)) if m else ("invalid", None, None)

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(p, 3), round(c - h, 3), round(c + h, 3)

def score(rows, outs, docs, k=3):
    bm = bm25_index(docs); M = {}
    def add(key, ok):
        a = M.setdefault(key, [0, 0]); a[0] += int(bool(ok)); a[1] += 1
    for x, out in zip(rows, outs):
        if x["stage"] == 1:
            dec, val = parse1(out); add("format_valid_s1", dec != "invalid")
            if x["type"] == "B":
                add("s1_direct_rate_on_B", dec == "direct")
                if dec == "direct": add("s1_B_answer_acc", norm(val) == norm(x["gold"]))
            else:
                add("s1_search_recall_on_ACD", dec == "search")
                if dec == "search" and x["type"] == "A":   # 검색어 평가: BM25 상위 k에 정답 문서 포함 여부
                    sc = bm.get_scores(TOK(val or "")); top = sorted(range(len(docs)), key=lambda i: -sc[i])[:k]
                    add(f"query_hit@{k}", any(docs[i]["entity"] == x["entity"] and x["attr"] in docs[i]["attrs"] for i in top))
        else:
            dec, val, cite = parse2(out); add("format_valid_s2", dec != "invalid")
            if x["gold"] is not None:
                add("s2_answer_acc", dec == "answer" and norm(val) == norm(x["gold"]))
                add("s2_false_abstain", dec == "abstain")
                if dec == "answer": add("s2_cite_acc", cite == x["gold_id"])
            else:
                add("s2_abstain_on_unanswerable", dec == "abstain")
            if x["distractor_ids"] and dec == "answer":
                add("s2_distractor_cited", cite in x["distractor_ids"])
    return {key: dict(zip(("rate", "ci_lo", "ci_hi"), wilson(*v)), n=v[1]) for key, v in M.items()}

if __name__ == "__main__":  # 자기 점검: 정답 그대로 넣으면 전부 1.0(유사 오답 인용률·거짓 보류는 0)
    rows = [json.loads(l) for f in ("data/val_stage1.jsonl", "data/val_stage2.jsonl") for l in open(f)]
    docs = [json.loads(l) for l in open("data/val_docs.jsonl")]
    print(json.dumps(score(rows, [x["target"] for x in rows], docs), indent=1))
    print(json.dumps(score(rows, ["ABSTAIN: insufficient_evidence"] * len(rows), docs)["s2_abstain_on_unanswerable"]))
