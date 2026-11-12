# run_full.py — HF Jobs용 본학습 러너. run_pilot.py와 같은 단계 0→4, 본 데이터(data_full/)로 실행.
# 실행(폴더를 Job에 마운트): hf jobs run --flavor l4x1 --timeout 3h -s HF_TOKEN -v ./pilot:/code ghcr.io/astral-sh/uv:python3.12-bookworm -- bash -c "cp -r /code /tmp/project && cd /tmp/project && uv run run_full.py"
import gc, json, os, sys, traceback

os.environ.setdefault("LEXSI_DATA_DIR", "data_full")
import app  # 모듈 로딩 시 Tiny Aya 로드 + cuda 배치
app.LOG = "/tmp/full_log.jsonl"; app.ADAPTER = "/tmp/full_adapter"


def fail(msg):
    print("FAIL:", msg, flush=True)
    sys.exit(1)


def step(name, fn):
    print(f"\n===== STEP {name} =====", flush=True)
    try:
        out = fn()
        print(out, flush=True)
        return out
    except Exception:
        traceback.print_exc()
        fail(f"step {name}")


def upload_artifacts(when):
    # 산출물 보존 시도 (실패해도 파일럿 자체는 성공으로 봄): 어댑터+로그를 비공개 Hub 데이터셋에 업로드
    print(f"\n===== UPLOAD ({when}, best effort) =====", flush=True)
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        repo = "baecha/lexsi-full"
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        for path, dest in [(app.ADAPTER, "adapter"), (app.LOG, "pilot_log.jsonl"),
                           ("/tmp/outs_full_baseline.json", "outs_full_baseline.json"),
                           ("/tmp/outs_full_adapter.json", "outs_full_adapter.json")]:
            if os.path.exists(path):
                api.upload_folder(folder_path=path, repo_id=repo, repo_type="dataset", path_in_repo=dest) \
                    if os.path.isdir(path) else api.upload_file(path_or_fileobj=path, path_in_repo=dest, repo_id=repo, repo_type="dataset")
        print("UPLOAD OK:", repo, flush=True)
    except Exception as e:
        print("UPLOAD SKIPPED/FAILED (어댑터는 Job 컨테이너 안에만 있음):", repr(e), flush=True)


# 0. 렌더링 확인 — AgentTune 학습 행과 평가 입력을 구조화 비교. 다르면 학습 전에 멈춤.
print("\n===== STEP 0 render-check =====", flush=True)
try:
    from agenttune import create_distill_trainer
    tr = create_distill_trainer(
        student=app.MODEL,
        demonstrations=[(x["prompt"], x["target"]) for x in app.TRAIN[:2]],
        output_dir="/tmp/render_check",
    )
    if not hasattr(tr, "sft_dataset"):
        fail("sft_dataset() 없음 — AgentTune 버전 확인 필요")
    row = tr.sft_dataset()[0]
    ref = app.chat(app.TRAIN[0]["prompt"])
    print("train row type:", type(row).__name__, flush=True)
    print("train row:", str(row)[:1500], flush=True)
    print("eval input:", ref, flush=True)
    match = None
    if isinstance(row, dict) and "messages" in row:
        # 학습 행이 메시지 형태면: user 턴만 채팅 템플릿으로 렌더링해 평가 입력과 비교
        user_msgs = [m for m in row["messages"] if m.get("role") == "user"]
        rendered = app.tok.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
        print("rendered train input:", rendered[:1500], flush=True)
        match = rendered.strip() == ref.strip()
    elif isinstance(row, dict):
        if "input_ids" in row:
            dec = app.tok.decode(row["input_ids"])
            print("decoded train row:", dec[:1500], flush=True)
            match = dec.strip() == ref.strip() or ref.strip() in dec
        elif "text" in row:
            match = row["text"].strip() == ref.strip()
        else:
            print("row keys:", list(row.keys()), flush=True)
    elif isinstance(row, str):
        match = row.strip() == ref.strip() or ref.strip() in row
    print("RENDER_MATCH:", match, flush=True)
    del tr
    gc.collect()
    if match is False:
        fail("render mismatch — stop before training")
except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    fail("render-check")

step("1 baseline", app.eval_baseline)
step("2 train", app.train)
if not os.path.exists(os.path.join(app.ADAPTER, "adapter_config.json")):
    fail("adapter not saved — see train log (saved_by/save_errors/diag)")
upload_artifacts("after-train")
step("3 adapter-check", app.adapter_check)
step("4 adapter-eval", app.eval_adapter)

print("\n===== FULL LOG =====", flush=True)
if os.path.exists(app.LOG):
    print(open(app.LOG).read(), flush=True)

upload_artifacts("final")
print("\nFULL DONE", flush=True)
