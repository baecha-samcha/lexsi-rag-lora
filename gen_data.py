# gen_data.py — 사실표 → 가상 문서 → 질의(A/B/C/D) → 1·2단계 예시(E1~E5). CPU만 씀.
import json, random, hashlib, re
from rank_bm25 import BM25Okapi

SEED = 13
TOK = lambda s: re.findall(r"[a-z0-9]+", s.lower())

# ---- 도메인 정의: 속성마다 (값 생성기, 문서 문장 틀, 질문 틀[학습용], 질문 틀[평가용], 검색어 단어)
# 도메인끼리 실체 이름·속성·값 집합·문장 틀이 안 겹치게 따로 씀(누수 방지).
DOMAINS = {
 "equipment_rental": dict(split="train", noun="an equipment item",
  names=["Halcyon Tripod","Vesper Projector","Quill Drone","Marlow Tent","Ostrander Kayak",
         "Pell Microscope","Tamsin Camera","Juno Speaker","Rook Generator","Fable Telescope"],
  attrs={
   "daily_fee": (lambda r: f"{r.randint(3,40)} credits", "Renting the {e} costs {v} per day.",
                 ["What is the daily fee for the {e}?","How much does the {e} cost per day?"],
                 ["Per day, what do I pay to rent the {e}?"], "daily fee"),
   "max_days": (lambda r: f"{r.randint(2,21)} days", "The {e} may be borrowed for at most {v}.",
                ["What is the maximum loan period for the {e}?","How long can I keep the {e}?"],
                ["For how many days at most can the {e} be borrowed?"], "maximum loan period"),
   "deposit": (lambda r: f"{r.choice([10,20,25,40,50,75,100])} credits", "A deposit of {v} is required for the {e}.",
               ["What deposit does the {e} require?","How big is the deposit for the {e}?"],
               ["Which deposit amount applies to the {e}?"], "deposit"),
   "pickup_desk": (lambda r: r.choice(["Desk North","Desk 4B","the Annex counter","Locker Bay 2","the Tool Cage"]),
                   "The {e} is collected at {v}.",
                   ["Where do I pick up the {e}?","At which desk is the {e} collected?"],
                   ["From which location is the {e} handed out?"], "pickup desk"),
   "late_fine": (lambda r: f"{r.randint(1,9)} credits per day", "Returning the {e} late incurs a fine of {v}.",
                 ["What is the late fine for the {e}?","How much is charged if the {e} is returned late?"],
                 ["What penalty applies to a late return of the {e}?"], "late fine"),
   "training_required": (lambda r: r.choice(["yes","no"]), "Mandatory safety training for the {e}: {v}.",
                 ["Is training required before using the {e}?","Do I need safety training for the {e}?"],
                 ["Must users complete safety training for the {e}?"], "safety training required"),
   "max_per_member": (lambda r: f"{r.randint(1,4)} units", "Each member may hold at most {v} of the {e}.",
                 ["How many units of the {e} can one member hold?"],
                 ["What is the per-member limit for the {e}?"], "per member limit"),
   "approval_by": (lambda r: r.choice(["the lab manager","the club treasurer","the facilities office","a senior member"]),
                 "Requests for the {e} are approved by {v}.",
                 ["Who approves requests for the {e}?"], ["Whose approval is needed for the {e}?"], "approval"),
  }),
 "facility_booking": dict(split="val", noun="a room",
  names=["Orion Hall","Lantern Studio","Birch Room","Cobalt Lab","Meridian Court",
         "Sable Lounge","Atlas Gym","Wren Gallery","Ember Kitchen","Tidewater Deck"],
  attrs={
   "capacity": (lambda r: f"{r.choice([8,12,16,24,30,40,60,120])} people", "{e} holds up to {v}.",
                ["What is the capacity of {e}?"], ["How many people fit in {e}?","Up to how many guests can {e} hold?"], "capacity"),
   "hourly_rate": (lambda r: f"{r.randint(5,60)} tokens", "Booking {e} costs {v} per hour.",
                ["What is the hourly rate of {e}?"], ["How much is one hour in {e}?","Per hour, what does {e} cost?"], "hourly rate"),
   "open_hours": (lambda r: r.choice(["07:00-19:00","08:30-22:00","10:00-18:00","09:00-23:30"]), "{e} is open {v}.",
                ["What are the opening hours of {e}?"], ["When is {e} open?","During which hours can {e} be used?"], "opening hours"),
   "advance_booking": (lambda r: f"{r.randint(1,14)} days", "{e} must be reserved at least {v} in advance.",
                ["How far in advance must {e} be booked?"], ["What is the minimum notice to reserve {e}?"], "advance booking notice"),
   "cancellation_window": (lambda r: f"{r.choice([6,12,24,48,72])} hours", "Bookings for {e} can be cancelled up to {v} before the start.",
                ["What is the cancellation window for {e}?"], ["Until when can a booking of {e} be cancelled?"], "cancellation window"),
   "key_holder": (lambda r: r.choice(["the porter","the events desk","Building Services","the duty warden"]), "Keys for {e} are kept by {v}.",
                ["Who holds the keys to {e}?"], ["Who keeps the key for {e}?"], "key holder"),
   "floor": (lambda r: r.choice(["ground floor","second floor","basement level","fourth floor"]), "{e} is located on the {v}.",
                ["On which floor is {e}?"], ["Which floor is {e} on?"], "floor location"),
   "food_allowed": (lambda r: r.choice(["yes","no"]), "Food permitted in {e}: {v}.",
                ["Is food allowed in {e}?"], ["May I bring food into {e}?"], "food allowed"),
  }),
}

def b_questions(r, n, pool):
    # B(비검색): 계산·상식. 절반은 코퍼스와 비슷한 어휘를 섞음(키워드만 보고 검색하는 습관 걸러내기).
    out = []
    for i in range(n):
        if i % 2 == 0:
            a, b = r.randint(3, 40), r.randint(2, 9)
            q = r.choice(pool["lookalike"]).format(a=a, b=b)
            out.append((q, str(a * b)))
        else:
            q, ans = r.choice(pool["general"])
            if callable(ans): q, ans = ans(r)
            out.append((q, ans))
    return out

B_POOLS = {
 "train": {"lookalike": ["If an item costs {a} credits per day, how many credits do {b} days cost?",
                         "A deposit of {a} credits is paid {b} times. What is the total in credits?"],
           "general": [(None, lambda r: (lambda x, y: (f"What is {x} plus {y}?", str(x + y)))(r.randint(10, 99), r.randint(10, 99))),
                       ("How many days are in a week?", "7"), ("How many hours are in a day?", "24")]},
 "val":   {"lookalike": ["A room costs {a} tokens per hour. How many tokens for {b} hours?",
                         "If {b} groups each have {a} people, how many people are there in total?"],
           "general": [(None, lambda r: (lambda x, y: (f"What is {x} minus {y}?", str(x - y)))(r.randint(50, 99), r.randint(1, 49))),
                       ("How many minutes are in an hour?", "60"), ("How many months are in a year?", "12")]},
}

def build_domain(dname, r):
    D = DOMAINS[dname]; attrs = list(D["attrs"]); docs, facts = [], {}
    for e in D["names"]:
        have = r.sample(attrs, 6)                # 실체마다 속성 8개 중 6개만 존재 → 나머지 2개가 C(답 없음) 재료
        facts[e] = {a: D["attrs"][a][0](r) for a in have}
        halves = [have[:3], have[3:]]            # 실체 하나 = 문서 2개(개요/규정). 형제 문서 = '같은 실체 다른 속성' 유사 오답
        for k, part in enumerate(halves):
            body = " ".join(D["attrs"][a][1].format(e=e, v=facts[e][a]) for a in part)
            filler = f" This page describes the {e}, {D['noun']} managed by the organization." \
                     f" Please read the general rules before making a request."
            docs.append(dict(doc_id=f"{dname[:3]}-{len(docs):03d}", entity=e, attrs=part,
                             title=f"{e} — {'overview' if k == 0 else 'policy'}", text=body + filler))
    return docs, facts

def render_stage1(q):
    return ("You are a help-desk assistant for an organization's internal documents.\n"
            "Tool: search_corpus(query: string) — searches the organization's documents.\n"
            "If answering needs the organization's documents, reply ONLY with:\n"
            '<tool_call>{"name": "search_corpus", "arguments": {"query": "<search words>"}}</tool_call>\n'
            "Otherwise reply ONLY with: ANSWER: <answer>\n\n"
            f"Question: {q}")

def render_stage2(q, bundle):
    docs = "\n\n".join(f"[{d['shown_id']}] {d['title']}\n{d['text']}" for d in bundle)
    return ("You are a help-desk assistant. Use ONLY the documents below.\n"
            "If the documents contain everything needed, reply: ANSWER: <answer> CITE: <doc id>\n"
            "If any needed information is missing, reply: ABSTAIN: insufficient_evidence\n\n"
            f"Question: {q}\n\nDocuments:\n{docs}")

def make_split(dname, r, n1, n2_plan, tmpl_key):
    D = DOMAINS[dname]; docs, facts = build_domain(dname, r)
    tk = 2 if tmpl_key == "train" else 3       # 학습/평가 질문 틀 분리
    s1, s2 = [], []
    ents = D["names"]
    def qtext(e, a): return r.choice(D["attrs"][a][tk]).format(e=e)
    def query(e, *a): return f"{e} " + " ".join(D["attrs"][x][4] for x in a)
    def doc_of(e, a): return next(d for d in docs if d["entity"] == e and a in d["attrs"])
    def unrelated(e, a, k):
        pool = [d for d in docs if d["entity"] != e and a not in d["attrs"]]
        return r.sample(pool, k)
    def similar(e, a):  # 유사 오답 2종 중 무작위: 다른 실체 같은 속성 / 같은 실체 다른 속성(형제 문서)
        other = [d for d in docs if d["entity"] != e and a in d["attrs"]]
        sib = [d for d in docs if d["entity"] == e and a not in d["attrs"]]
        kind = r.choice(["other_entity", "sibling"]) if other and sib else ("other_entity" if other else "sibling")
        return (r.choice(other) if kind == "other_entity" else sib[0]), kind
    # --- 1단계 질의: A/C/D는 검색, B는 직접 답변. A와 C는 같은 질문 틀 사용.
    cnt = n1
    for typ, k in cnt.items():
        for _ in range(k):
            if typ == "B": continue
            e = r.choice(ents); present = list(facts[e]); absent = [a for a in D["attrs"] if a not in facts[e]]
            if typ == "A": a = r.choice(present); s1.append(dict(type="A", entity=e, attr=a, q=qtext(e, a), query=query(e, a), gold=facts[e][a]))
            if typ == "C": a = r.choice(absent); s1.append(dict(type="C", entity=e, attr=a, q=qtext(e, a), query=query(e, a), gold=None))
            if typ == "D":
                a1, a2 = r.choice(present), r.choice(absent)
                q = f"What are the {D['attrs'][a1][4]} and the {D['attrs'][a2][4]} of the {e}?" if tmpl_key == "train" \
                    else f"Tell me both the {D['attrs'][a1][4]} and the {D['attrs'][a2][4]} for the {e}."
                s1.append(dict(type="D", entity=e, attr=a1, attr2=a2, q=q, query=query(e, a1, a2), gold=None))
    for q, ans in b_questions(r, cnt.get("B", 0), B_POOLS[tmpl_key]):
        s1.append(dict(type="B", q=q, gold=ans))
    for x in s1:
        x["stage"] = 1; x["domain"] = dname; x["prompt"] = render_stage1(x["q"])
        x["target"] = (f'<tool_call>{json.dumps({"name": "search_corpus", "arguments": {"query": x["query"]}})}</tool_call>'
                       if x["type"] != "B" else f"ANSWER: {x['gold']}")
    # --- 2단계: 질문+문서 3개 묶음. 라벨은 묶음 기준.
    for (typ, ev), k in n2_plan.items():
        for _ in range(k):
            e = r.choice(ents); present = list(facts[e]); absent = [a for a in D["attrs"] if a not in facts[e]]
            a = r.choice(present if typ in ("A", "D") else absent)
            gold_doc, dkind = None, None
            if ev == "E1": gold_doc = doc_of(e, a); bundle = [gold_doc] + unrelated(e, a, 2)
            if ev == "E2": gold_doc = doc_of(e, a); sim, dkind = similar(e, a); bundle = [gold_doc, sim] + unrelated(e, a, 1)
            if ev == "E3": sim, dkind = similar(e, a); bundle = [sim] + unrelated(e, a, 2)
            if ev == "E4": bundle = unrelated(e, a, 3)
            if ev == "E5":                        # D: 슬롯 1개만 있는 문서
                a2 = r.choice(absent); bundle = [doc_of(e, a)] + unrelated(e, a, 2)
            r.shuffle(bundle)                     # 정답 문서 위치 무작위
            ids = r.sample(range(10, 99), 3)      # 문서 번호도 묶음마다 무작위
            shown = [dict(d, shown_id=f"d{i}") for d, i in zip(bundle, ids)]
            if typ == "D":
                q = f"What are the {D['attrs'][a][4]} and the {D['attrs'][a2][4]} of the {e}?" if tmpl_key == "train" \
                    else f"Tell me both the {D['attrs'][a][4]} and the {D['attrs'][a2][4]} for the {e}."
            else:
                q = qtext(e, a)
            answerable = gold_doc is not None
            gid = next((s["shown_id"] for s in shown if gold_doc and s["doc_id"] == gold_doc["doc_id"]), None)
            dist_ids = [s["shown_id"] for s in shown if sim_match(s, e, a)] if ev in ("E2", "E3") else []
            x = dict(stage=2, domain=dname, type=typ, evidence=ev, distractor_kind=dkind, entity=e, attr=a, q=q,
                     bundle=[s["doc_id"] for s in shown], shown_ids=[s["shown_id"] for s in shown],
                     gold=facts[e][a] if answerable else None, gold_id=gid, distractor_ids=dist_ids,
                     prompt=render_stage2(q, shown))
            x["target"] = f"ANSWER: {x['gold']} CITE: {gid}" if answerable else "ABSTAIN: insufficient_evidence"
            s2.append(x)
    return docs, facts, s1, s2

def sim_match(s, e, a):  # 유사 오답 문서 식별: 다른 실체·같은 속성 또는 같은 실체·다른 속성
    return (s["entity"] != e and a in s["attrs"]) or (s["entity"] == e and a not in s["attrs"])

def bm25_index(docs):
    return BM25Okapi([TOK(d["title"] + " " + d["text"]) for d in docs])

# 파일럿 규모: 1단계 40 (A18 B10 C8 D4) + 2단계 60 (답변 28 : 보류 32)
PILOT_S1 = {"A": 18, "B": 10, "C": 8, "D": 4}
PILOT_S2 = {("A", "E1"): 14, ("A", "E2"): 14, ("A", "E3"): 10, ("C", "E3"): 5, ("C", "E4"): 6, ("D", "E5"): 11}

def main(out="data"):
    import os; os.makedirs(out, exist_ok=True)
    r = random.Random(SEED)
    for dname, tmpl in [("equipment_rental", "train"), ("facility_booking", "val")]:
        docs, facts, s1, s2 = make_split(dname, r, PILOT_S1, PILOT_S2, tmpl)
        split = DOMAINS[dname]["split"]
        for name, rows in [("docs", docs), ("stage1", s1), ("stage2", s2)]:
            with open(f"{out}/{split}_{name}.jsonl", "w") as f:
                for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        json.dump(facts, open(f"{out}/{split}_facts.json", "w"), indent=1)
    # 누수 검사: 학습/검증 사이 실체 이름·질문 문자열 겹침 0이어야 함
    tr = [json.loads(l) for l in open(f"{out}/train_stage1.jsonl")] + [json.loads(l) for l in open(f"{out}/train_stage2.jsonl")]
    va = [json.loads(l) for l in open(f"{out}/val_stage1.jsonl")] + [json.loads(l) for l in open(f"{out}/val_stage2.jsonl")]
    assert not ({x["q"] for x in tr} & {x["q"] for x in va}), "question overlap"
    assert not (set(DOMAINS["equipment_rental"]["names"]) & set(DOMAINS["facility_booking"]["names"]))
    h = hashlib.sha256(open(f"{out}/val_stage2.jsonl", "rb").read()).hexdigest()
    print("ok", len(tr), "train /", len(va), "val | val_stage2 sha256", h[:16])

if __name__ == "__main__":
    main()
