# eval_test.py — 분리 테스트(시험 세트) 1회 평가. 베이스라인 + 본학습 어댑터를 시험 세트에서 평가.
# 실행(Job): hf jobs run --flavor l4x1 --timeout 1h -s HF_TOKEN -v./pilot:/code ghcr.io/astral-sh/uv:python3.12-bookworm -- bash -c "cp -r /code /tmp/project && cd /tmp/project && uv run eval_test.py"
# 주의: 시험 세트는 최종 1회만 쓴다. 이 스크립트를 돌리면 시험 세트는 사용된 것으로 본다.
# 시험 데이터는 Hub 공개 데이터셋(baecha/lexsi-rag-data)에 포함돼 있다. 제출용 평가는 1회만 수행했고, 이 스크립트를 다시 돌리면 2회 사용이 된다.
import json, os, time
os.environ.setdefault("LEXSI_DATA_DIR", "data_full")
import app
from score import score as score_fn
from peft import PeftModel
from huggingface_hub import snapshot_download

TEST = app.load(f"{app.DATA_DIR}/test_stage1.jsonl") + app.load(f"{app.DATA_DIR}/test_stage2.jsonl")
TEST_DOCS = app.load(f"{app.DATA_DIR}/test_docs.jsonl")

def main():
    t = time.perf_counter()
    outs_base = app.generate(app.base, TEST)
    m_base = score_fn(TEST, outs_base, TEST_DOCS)
    json.dump(outs_base, open("/tmp/outs_test_baseline.json", "w"))
    print("TEST BASELINE:", json.dumps(m_base, indent=1), flush=True)
    path = snapshot_download(repo_id="baecha/lexsi-full", repo_type="dataset",
                             allow_patterns=["adapter/*"], local_dir="/tmp/test_adapter_dl")
    model = PeftModel.from_pretrained(app.base, os.path.join(path, "adapter"))
    outs_ad = app.generate(model, TEST)
    m_ad = score_fn(TEST, outs_ad, TEST_DOCS)
    json.dump(outs_ad, open("/tmp/outs_test_adapter.json", "w"))
    print("TEST ADAPTER:", json.dumps(m_ad, indent=1), flush=True)
    print(f"gpu_sec={time.perf_counter() - t:.1f}", flush=True)
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for f in ["/tmp/outs_test_baseline.json", "/tmp/outs_test_adapter.json"]:
            api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f),
                            repo_id="baecha/lexsi-full", repo_type="dataset")
        print("UPLOAD OK: baecha/lexsi-full", flush=True)
    except Exception as e:
        print("UPLOAD FAILED:", repr(e), flush=True)
    print("TEST DONE — 시험 세트 사용 완료 (재평가 금지: 누수 방지)", flush=True)

if __name__ == "__main__":
    main()
