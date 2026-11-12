# gen_data_full.py — 본 데이터 생성 (Phase 2 설계). 파일럿 gen_data.py의 생성기를 재사용.
# 실행: pilot 폴더 안에서 python3 gen_data_full.py  (rank_bm25 필요: pip install rank_bm25)
# 출력: data_full/ — train(4도메인) / val(1도메인) / test(1도메인) + 손글 세트 안내 + manifest.json
# 나무위키는 학습에 쓰지 않음 (Phase 3 권고대로 일반화 평가 세트로만 분리 — 별도 수작업).
import json, random, hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_data as pg

SEED = 29

# ---- 신규 도메인 4개: 실체 이름·속성·값·문장 틀·질문 틀이 기존 도메인과 안 겹치게 따로 씀.
NEW_DOMAINS = {
 "vehicle_checkout": dict(split="train", noun="a vehicle",
  names=["Harbor Van", "Juniper Sedan", "Alder Truck", "Briar Minibus", "Cinder Coupe",
         "Dune Pickup", "Elm Shuttle", "Fjord Estate", "Grove Hatchback", "Heath Convertible"],
  attrs={
   "daily_rate": (lambda r: f"{r.randint(20, 120)} coins", "Renting the {e} costs {v} per day.",
                 ["What is the daily rate for the {e}?", "How much does the {e} cost per day?"],
                 ["Per day, what do I pay to rent the {e}?"], "daily rate"),
   "seats": (lambda r: f"{r.choice([2, 4, 5, 7, 9, 12, 15])} seats", "The {e} seats {v}.",
                ["How many seats does the {e} have?"],
                ["How many people can sit in the {e}?"], "seats"),
   "fuel_type": (lambda r: r.choice(["petrol", "diesel", "electric", "hybrid"]), "The {e} runs on {v}.",
                ["What fuel does the {e} use?"], ["Which fuel type does the {e} take?"], "fuel type"),
   "pickup_lot": (lambda r: r.choice(["Lot A", "Lot C", "the North Gate", "the Depot", "Bay 7"]),
                   "The {e} is collected from {v}.",
                   ["Where do I collect the {e}?", "From which lot is the {e} picked up?"],
                   ["At which location is the {e} handed over?"], "pickup lot"),
   "mileage_limit": (lambda r: f"{r.choice([100, 150, 200, 300])} km per day", "The {e} may be driven up to {v}.",
                ["What is the mileage limit for the {e}?"],
                ["How far per day may the {e} be driven?"], "mileage limit"),
   "insurance_included": (lambda r: r.choice(["yes", "no"]), "Insurance for the {e} is included: {v}.",
                ["Is insurance included for the {e}?"],
                ["Does the rental of the {e} include insurance?"], "insurance included"),
   "booking_lead": (lambda r: f"{r.randint(1, 7)} days", "The {e} must be booked at least {v} in advance.",
                ["How far in advance must the {e} be booked?"],
                ["What is the minimum notice to book the {e}?"], "booking lead"),
   "driver_required": (lambda r: r.choice(["yes", "no"]), "A licensed driver is required for the {e}: {v}.",
                ["Is a licensed driver required for the {e}?"],
                ["Must a licensed driver drive the {e}?"], "licensed driver required"),
  }),
 "catering_order": dict(split="train", noun="a catering package",
  names=["Amber Buffet", "Basil Box", "Cedar Feast", "Dill Platter", "Embers Grill",
         "Fig Table", "Ginger Spread", "Honey Board", "Ivy Banquet", "Juniper Picnic"],
  attrs={
   "price_per_person": (lambda r: f"{r.randint(8, 45)} coins", "The {e} costs {v} per person.",
                ["What is the price per person for the {e}?", "How much is the {e} per person?"],
                ["Per person, what does the {e} cost?"], "price per person"),
   "min_order": (lambda r: f"{r.choice([10, 15, 20, 25, 30, 50])} people", "The {e} requires a minimum order for {v}.",
                ["What is the minimum order for the {e}?"],
                ["How many people at least must the {e} serve?"], "minimum order"),
   "lead_time": (lambda r: f"{r.randint(2, 10)} days", "The {e} must be ordered at least {v} in advance.",
                ["How far in advance must the {e} be ordered?"],
                ["What is the minimum notice to order the {e}?"], "lead time"),
   "dietary_options": (lambda r: r.choice(["vegetarian", "vegan", "halal", "gluten-free"]), "The {e} offers a {v} option.",
                ["Which dietary option does the {e} offer?"],
                ["What dietary choice is available with the {e}?"], "dietary option"),
   "delivery_fee": (lambda r: f"{r.randint(0, 25)} coins", "Delivery for the {e} costs {v}.",
                ["What is the delivery fee for the {e}?"],
                ["How much is delivery for the {e}?"], "delivery fee"),
   "setup_included": (lambda r: r.choice(["yes", "no"]), "Setup is included with the {e}: {v}.",
                ["Is setup included with the {e}?"], ["Does the {e} come with setup?"], "setup included"),
   "cancellation_window": (lambda r: f"{r.choice([24, 48, 72])} hours", "The {e} can be cancelled up to {v} before the event.",
                ["What is the cancellation window for the {e}?"],
                ["Until when can the {e} be cancelled?"], "cancellation window"),
   "contact_person": (lambda r: r.choice(["the kitchen manager", "the events team", "the catering desk", "the head chef"]),
                "Questions about the {e} go to {v}.",
                ["Who do I contact about the {e}?"], ["Whom should I ask about the {e}?"], "contact person"),
  }),
 "software_license": dict(split="train", noun="a software tool",
  names=["Atlas Notes", "Beacon Draw", "Cobalt Calc", "Drift Mail", "Echo Slides",
         "Flint Code", "Glide Board", "Harbor Wiki", "Iris Forms", "Jetty Chat"],
  attrs={
   "seats": (lambda r: f"{r.choice([5, 10, 25, 50, 100])} seats", "The {e} licence covers {v}.",
                ["How many seats does the {e} licence cover?"],
                ["What is the seat count for the {e}?"], "licence seats"),
   "cost_per_seat": (lambda r: f"{r.randint(3, 40)} coins", "Each seat of the {e} costs {v} per month.",
                ["What is the cost per seat for the {e}?"],
                ["Per seat per month, what does the {e} cost?"], "cost per seat"),
   "renewal_date": (lambda r: r.choice(["1 March", "15 June", "30 September", "1 December"]), "The {e} licence renews on {v}.",
                ["When does the {e} licence renew?"], ["On which date does the {e} renew?"], "renewal date"),
   "admin_contact": (lambda r: r.choice(["the IT desk", "the licence manager", "the software team", "the admin office"]),
                "Licence questions for the {e} go to {v}.",
                ["Who manages the licence for the {e}?"], ["Whom do I ask about the {e} licence?"], "licence manager"),
   "install_method": (lambda r: r.choice(["self-service portal", "IT deployment", "download link", "usb installer"]),
                "The {e} is installed via {v}.",
                ["How is the {e} installed?"], ["By which method is the {e} installed?"], "install method"),
   "support_hours": (lambda r: r.choice(["09:00-17:00", "08:00-20:00", "24 hours"]), "Support for the {e} is available {v}.",
                ["What are the support hours for the {e}?"],
                ["When is support for the {e} available?"], "support hours"),
   "current_version": (lambda r: f"{r.randint(2, 9)}.{r.randint(0, 9)}", "The current version of the {e} is {v}.",
                ["What is the current version of the {e}?"], ["Which version is the {e} on now?"], "current version"),
   "platform": (lambda r: r.choice(["Windows", "macOS", "Linux", "web only"]), "The {e} runs on {v}.",
                ["Which platform does the {e} run on?"], ["On what platform does the {e} run?"], "platform"),
  }),
 "parking_permit": dict(split="test", noun="a parking area",
  names=["North Yard", "East Deck", "South Court", "West Bay", "Cedar Row",
         "Maple Terrace", "Birch Lane", "Elm Plaza", "Willow Court", "Ash Field"],
  attrs={
   "hourly_fee": (lambda r: f"{r.randint(1, 8)} coins", "Parking in {e} costs {v} per hour.",
                ["What is the hourly fee for {e}?"], ["Per hour, what does parking in {e} cost?"], "hourly fee"),
   "monthly_fee": (lambda r: f"{r.randint(20, 120)} coins", "A monthly pass for {e} costs {v}.",
                ["What is the monthly fee for {e}?"], ["How much is a monthly pass for {e}?"], "monthly fee"),
   "max_stay": (lambda r: f"{r.choice([2, 4, 8, 12, 24])} hours", "Vehicles may stay in {e} for at most {v}.",
                ["What is the maximum stay in {e}?"], ["How long at most can a vehicle stay in {e}?"], "maximum stay"),
   "permit_required": (lambda r: r.choice(["yes", "no"]), "A permit is required for {e}: {v}.",
                ["Is a permit required for {e}?"], ["Must drivers hold a permit for {e}?"], "permit required"),
   "ev_charging": (lambda r: r.choice(["yes", "no"]), "EV charging is available in {e}: {v}.",
                ["Is EV charging available in {e}?"], ["Does {e} offer EV charging?"], "EV charging"),
   "disabled_bays": (lambda r: f"{r.randint(1, 6)} bays", "{e} has {v} for disabled drivers.",
                ["How many disabled bays does {e} have?"],
                ["What is the number of disabled bays in {e}?"], "disabled bays"),
   "open_hours": (lambda r: r.choice(["06:00-23:00", "00:00-24:00", "07:00-21:00"]), "{e} is open {v}.",
                ["What are the opening hours of {e}?"], ["When is {e} open?"], "opening hours"),
   "shuttle_service": (lambda r: r.choice(["yes", "no"]), "A shuttle serves {e}: {v}.",
                ["Is there a shuttle service for {e}?"], ["Does a shuttle run to {e}?"], "shuttle service"),
  }),
}

# ---- 문서 길이 보강: 파일럿 문서(~60단어) → 본 데이터 100~250단어. 사실이 없는 중립 문장만 씀.
FILLER = [
 "This page describes the {e}, {noun} managed by the organization.",
 "Please read the general rules before making a request.",
 "If anything on this page looks out of date, tell the office.",
 "The office reviews these pages during the annual update.",
 "A printed copy of this page is kept at the front desk.",
 "Questions about this page go to the help desk.",
 "This page is part of the organization's internal guide.",
 "Only members of the organization may use the {e}.",
 "The {e} is listed in the organization's catalogue.",
 "See the notice board near the entrance for general announcements.",
 "Members should keep this page for their own reference.",
 "The help desk can explain any part of this page.",
]

B_POOLS_TEST = {
 "lookalike": ["A parking area charges {a} coins per hour. How many coins for {b} hours?",
               "If {b} cars each pay {a} coins, what is the total in coins?"],
 "general": [(None, lambda r: (lambda x, y: (f"What is {x * y} divided by {y}?", str(x)))(r.randint(3, 12), r.randint(2, 9))),
             ("How many seconds are in a minute?", "60"), ("How many weeks are in a year?", "52")],
}

def build_domain_full(dname, r):
    D = pg.DOMAINS[dname]; attrs = list(D["attrs"]); docs, facts = [], {}
    for e in D["names"]:
        have = r.sample(attrs, 6)
        facts[e] = {a: D["attrs"][a][0](r) for a in have}
        halves = [have[:3], have[3:]]
        for k, part in enumerate(halves):
            body = " ".join(D["attrs"][a][1].format(e=e, v=facts[e][a]) for a in part)
            filler = " ".join(s.format(e=e, noun=D["noun"]) for s in r.sample(FILLER, 9))
            docs.append(dict(doc_id=f"{dname[:3]}-{len(docs):03d}", entity=e, attrs=part,
                             title=f"{e} — {'overview' if k == 0 else 'policy'}", text=body + " " + filler))
    return docs, facts

# ---- 규모 (Phase 2 설계): 학습 1단계 500개(A225/B125/C100/D50) + 2단계 824개 = 1,324개
FULL_S1_TRAIN = {"A": 56, "B": 31, "C": 25, "D": 13}                       # 도메인당 125개 × 4 = 500
FULL_S2_TRAIN = {("A", "E1"): 48, ("A", "E2"): 48, ("A", "E3"): 34,
                 ("C", "E3"): 17, ("C", "E4"): 21, ("D", "E5"): 38}       # 도메인당 206개 × 4 = 824
TEST_S1 = {"A": 27, "B": 15, "C": 12, "D": 6}                             # 60개
TEST_S2 = {("A", "E1"): 21, ("A", "E2"): 21, ("A", "E3"): 15,
           ("C", "E3"): 8, ("C", "E4"): 9, ("D", "E5"): 16}               # 90개

HANDWRITTEN_README = """# 손글 세트 (80개) — 사람이 직접 작성, 모델 결과를 보기 전에 해시로 고정
- 시험 세트와 별개로, 템플릿 누수를 피하려고 사람이 손으로 쓰는 80개 질의.
- 작성 요령: 가상 문서 도메인과 겹치지 않는 주제/표현으로, A(답 있음)/C(답 없음)를 같은 질문 틀로.
- 고정: 작성 완료 후 `sha256sum test_handwritten.jsonl`을 기록하고, 최종 평가 1회만 사용.
- 파일: test_handwritten.jsonl (stage1/stage2와 같은 스키마: stage, domain, type, q, prompt, target, gold, gold_id, distractor_ids)
"""

def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

def write(out, name, rows):
    p = f"{out}/{name}"
    with open(p, "w") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    return p

def main(out="data_full"):
    os.makedirs(out, exist_ok=True)
    pg.DOMAINS = {**pg.DOMAINS, **NEW_DOMAINS}
    pg.B_POOLS["test"] = B_POOLS_TEST
    pg.build_domain = build_domain_full
    r = random.Random(SEED)
    tr_docs, tr_facts, tr_s1, tr_s2 = [], {}, [], []
    for dname in ["equipment_rental", "vehicle_checkout", "catering_order", "software_license"]:
        docs, facts, s1, s2 = pg.make_split(dname, r, FULL_S1_TRAIN, FULL_S2_TRAIN, "train")
        tr_docs += docs; tr_facts.update(facts); tr_s1 += s1; tr_s2 += s2
    va_docs, va_facts, va_s1, va_s2 = pg.make_split("facility_booking", r, pg.PILOT_S1, pg.PILOT_S2, "val")
    te_docs, te_facts, te_s1, te_s2 = pg.make_split("parking_permit", r, TEST_S1, TEST_S2, "test")
    files = {}
    for split, docs, facts, s1, s2 in [("train", tr_docs, tr_facts, tr_s1, tr_s2),
                                       ("val", va_docs, va_facts, va_s1, va_s2),
                                       ("test", te_docs, te_facts, te_s1, te_s2)]:
        files[f"{split}_docs"] = write(out, f"{split}_docs.jsonl", docs)
        files[f"{split}_stage1"] = write(out, f"{split}_stage1.jsonl", s1)
        files[f"{split}_stage2"] = write(out, f"{split}_stage2.jsonl", s2)
        json.dump(facts, open(f"{out}/{split}_facts.json", "w"), indent=1)
        files[f"{split}_facts"] = f"{out}/{split}_facts.json"
    open(f"{out}/handwritten_README.md", "w").write(HANDWRITTEN_README)
    # ---- 검증: 개수·누수·문서 길이
    assert len(tr_s1) == 500 and len(tr_s2) == 824, (len(tr_s1), len(tr_s2))
    assert len(va_s1) == 40 and len(va_s2) == 60
    assert len(te_s1) == 60 and len(te_s2) == 90
    assert len(tr_docs) == 80 and len(va_docs) == 20 and len(te_docs) == 20
    qs = {s: {x["q"] for x in rows} for s, rows in
          [("train", tr_s1 + tr_s2), ("val", va_s1 + va_s2), ("test", te_s1 + te_s2)]}
    assert not (qs["train"] & qs["val"]) and not (qs["train"] & qs["test"]) and not (qs["val"] & qs["test"]), "question overlap"
    names = {d: set(pg.DOMAINS[d]["names"]) for d in pg.DOMAINS}
    alln = list(names.values())
    for i in range(len(alln)):
        for j in range(i + 1, len(alln)):
            assert not (alln[i] & alln[j]), "entity overlap"
    wc = {s: sorted(len(d["text"].split()) for d in docs) for s, docs in
          [("train", tr_docs), ("val", va_docs), ("test", te_docs)]}
    for s, w in wc.items():
        assert 100 <= w[0], f"{s} doc too short: {w[0]} words"
        assert w[-1] <= 250, f"{s} doc too long: {w[-1]} words"
    manifest = {"seed": SEED,
                "counts": {"train_stage1": len(tr_s1), "train_stage2": len(tr_s2),
                           "val_stage1": len(va_s1), "val_stage2": len(va_s2),
                           "test_stage1": len(te_s1), "test_stage2": len(te_s2),
                           "train_docs": len(tr_docs), "val_docs": len(va_docs), "test_docs": len(te_docs)},
                "doc_words": {s: {"min": w[0], "median": w[len(w) // 2], "max": w[-1]} for s, w in wc.items()},
                "sha256": {k: sha(p)[:16] for k, p in files.items()}}
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
    print("ok train", len(tr_s1) + len(tr_s2), "/ val", len(va_s1) + len(va_s2), "/ test", len(te_s1) + len(te_s2))
    print("doc words:", {s: (w[0], w[len(w) // 2], w[-1]) for s, w in wc.items()})
    print("test_stage2 sha256:", sha(files["test_stage2"])[:16], "(최종 평가 전까지 건드리지 말 것)")

if __name__ == "__main__":
    main()
