# app.py — HF ZeroGPU Space (Gradio SDK, Python 3.12.12). 버튼: 렌더링 확인 / 베이스라인 평가 / 학습 / 어댑터 점검 / 학습 후 평가
import os, json, time, glob, torch
try:
    import spaces; GPU = lambda d: spaces.GPU(duration=d)
except ImportError:                                   # Colab·Kaggle에서도 그대로 돌게
    GPU = lambda d: (lambda f: f)
from transformers import AutoTokenizer, AutoModelForCausalLM
import gen_data, score

MODEL = "CohereLabs/tiny-aya-global"
ADAPTER = "/tmp/pilot_adapter"; LOG = "/tmp/pilot_log.jsonl"
DATA_DIR = os.environ.get("LEXSI_DATA_DIR", "data")
if not os.path.exists(f"{DATA_DIR}/val_stage2.jsonl"):
    if DATA_DIR == "data": gen_data.main("data")
    else: raise SystemExit(f"DATA_DIR={DATA_DIR}에 데이터 없음 — gen_data_full.py를 먼저 실행하세요")
load = lambda p: [json.loads(l) for l in open(p)]
TRAIN = load(f"{DATA_DIR}/train_stage1.jsonl") + load(f"{DATA_DIR}/train_stage2.jsonl")
VAL = load(f"{DATA_DIR}/val_stage1.jsonl") + load(f"{DATA_DIR}/val_stage2.jsonl")  # 검증 도메인, 시험 세트 아님
VAL_DOCS = load(f"{DATA_DIR}/val_docs.jsonl")

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
# Tiny Aya turn 종료 토큰에서도 멈추게 함 (기본 EOS만으로는 안 멈춤)
_EOT = tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")
_STOP_IDS = sorted({i for i in [tok.eos_token_id, _EOT] if isinstance(i, int)})
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
base.to("cuda")                                       # ZeroGPU 권장: 모듈 최상단에서 cuda 배치

def chat(prompt):  # 학습·평가 공통 렌더링. 학습 쪽 렌더링과 문자열이 같은지 check_render()로 확인
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

def log(rec):
    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S"); open(LOG, "a").write(json.dumps(rec) + "\n"); return rec

def generate(model, rows, bs=16, max_new=64):
    outs = []
    model.eval()
    for i in range(0, len(rows), bs):
        enc = tok([chat(x["prompt"]) for x in rows[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=_STOP_IDS)   # 탐욕적 생성 고정
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs

@GPU(110)
def eval_baseline():
    t = time.perf_counter(); outs = generate(base, VAL)
    m = score.score(VAL, outs, VAL_DOCS)
    json.dump(outs, open("/tmp/outs_baseline.json", "w"))
    return json.dumps(log(dict(step="baseline", gpu_sec=round(time.perf_counter() - t, 1),
                               peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1), metrics=m)), indent=1)

def check_render():
    # GPU 불필요. AgentTune이 만드는 학습 행과 우리 평가 입력을 나란히 출력 → 같지 않으면 학습 전에 멈출 것
    from agenttune import create_distill_trainer
    tr = create_distill_trainer(student=MODEL, demonstrations=[(x["prompt"], x["target"]) for x in TRAIN[:2]],
                                output_dir="/tmp/render_check")
    row = tr.sft_dataset()[0] if hasattr(tr, "sft_dataset") else "sft_dataset() 없음 — AgentTune 버전 확인"
    return f"=== AgentTune 학습 행 ===\n{row}\n\n=== 평가 입력 ===\n{chat(TRAIN[0]['prompt'])}"

@GPU(120)
def train():
    from agenttune import create_distill_trainer
    t = time.perf_counter()
    tr = create_distill_trainer(
        student=MODEL,
        demonstrations=[(x["prompt"], x["target"]) for x in TRAIN],
        output_dir="/tmp/out_distill",
        use_peft=True, lora_r=16, lora_alpha=32, lora_dropout=0.05,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],   # Tiny Aya 모듈 이름은 첫 실행에서 확인(adapter_check)
        num_train_epochs=3, learning_rate=2e-4, per_device_train_batch_size=4,
        bf16=True, save_strategy="epoch", report_to="none",  # DistillTrainer는 모델 핸들을 안 줘서 체크포인트로 어댑터를 꺼냄
    )
    res = tr.train()
    # --- 어댑터 저장: 이전 실행에서 두 저장 시도 모두 조용히 실패해 adapter_config.json이 없었음.
    # 각 시도의 예외를 기록하고, adapter_config.json이 있는 저장만 인정. PeftModel을 먼저 찾아서 어댑터만 저장.
    import shutil
    diag = {"trainer_type": type(tr).__name__}
    for attr in ("model", "student", "trainer", "peft_model"):
        if hasattr(tr, attr):
            diag[attr] = type(getattr(tr, attr)).__name__
    diag["trainer_attrs"] = [a for a in dir(tr)
                            if any(k in a.lower() for k in ("model", "save", "student", "peft", "adapter"))]
    peft_model = None
    try:
        from peft import PeftModel
        for v in list(vars(tr).values()) + [getattr(tr, "model", None)]:
            if isinstance(v, PeftModel):
                peft_model = v
                break
        diag["peft_model_found"] = peft_model is not None
    except Exception as e:
        diag["peft_search_error"] = repr(e)
    attempts = []
    if peft_model is not None:
        attempts.append(("peft_model.save_pretrained", lambda d: peft_model.save_pretrained(d)))
    if hasattr(tr, "save_model"):
        attempts.append(("tr.save_model", lambda d: tr.save_model(d)))
    saved, errors = None, {}
    for i, (name, fn) in enumerate(attempts):
        tmp = f"/tmp/adapter_try{i}"
        try:
            fn(tmp)
            if os.path.exists(os.path.join(tmp, "adapter_config.json")):
                shutil.rmtree(ADAPTER, ignore_errors=True)
                shutil.copytree(tmp, ADAPTER)
                saved = name
                break
            errors[name] = "saved but no adapter_config.json; files=" + str(os.listdir(tmp)[:10])
        except Exception as e:
            errors[name] = repr(e)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    found = glob.glob("/tmp/out_distill/**/adapter_config.json", recursive=True)
    if saved is None and found:  # 체크포인트 저장분 (가장 늦은 체크포인트의 어댑터)
        def _ckpt(p):
            tail = p.split("checkpoint-")[-1].split("/")[0] if "checkpoint-" in p else "-1"
            return int(tail) if tail.isdigit() else -1
        src = os.path.dirname(max(found, key=_ckpt))
        shutil.rmtree(ADAPTER, ignore_errors=True)
        shutil.copytree(src, ADAPTER)
        saved = f"copied from {src}"
    return json.dumps(log(dict(step="train", gpu_sec=round(time.perf_counter() - t, 1), saved_by=saved,
                               save_errors=errors, diag=diag,
                               adapter_files=os.listdir(ADAPTER) if os.path.exists(ADAPTER) else None,
                               found_in_output_dir=found,
                               train_result=str(res)[:500])), indent=1)

@GPU(110)
def eval_adapter():
    from peft import PeftModel
    t = time.perf_counter()
    model = PeftModel.from_pretrained(base, ADAPTER)
    outs = generate(model, VAL); m = score.score(VAL, outs, VAL_DOCS)
    json.dump(outs, open("/tmp/outs_adapter.json", "w"))
    model.unload()                                    # base를 원래대로 되돌림
    return json.dumps(log(dict(step="adapter", gpu_sec=round(time.perf_counter() - t, 1), metrics=m)), indent=1)

@GPU(60)
def adapter_check():
    # GPT·Claude 합의 기준 (a)~(c): lora_B 비영, 활성 어댑터, 켜고 끈 로그확률 차이
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, ADAPTER); model.eval()
    lb = [p.abs().max().item() for n, p in model.named_parameters() if "lora_B" in n]
    enc = tok([chat(VAL[0]["prompt"])], return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        on = model(**enc).logits[0, -1].float().log_softmax(-1)
        with model.disable_adapter(): off = model(**enc).logits[0, -1].float().log_softmax(-1)
    st = model.get_model_status()
    model.unload()
    mods = sorted({n.split(".")[-1] for n, _ in base.named_modules() if n.endswith("proj")})
    return json.dumps(log(dict(step="adapter_check", n_lora_B=len(lb), lora_B_max=max(lb) if lb else None,
                               lora_B_all_zero=all(v == 0 for v in lb), active=str(st.active_adapters),
                               max_logprob_diff=round((on - off).abs().max().item(), 4), proj_modules=mods)), indent=1)

if __name__ == "__main__":
    import gradio as gr
    with gr.Blocks() as demo:
        out = gr.Code(language="json")
        for label, fn in [("0. 렌더링 확인(CPU)", check_render), ("1. 베이스라인 평가", eval_baseline), ("2. 학습", train),
                          ("3. 어댑터 점검", adapter_check), ("4. 학습 후 평가", eval_adapter)]:
            gr.Button(label).click(fn, outputs=out)
        gr.Button("로그 보기").click(lambda: open(LOG).read() if os.path.exists(LOG) else "(없음)", outputs=out)
    demo.launch()
