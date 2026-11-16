"""번역 엔진: Gemma 4 E2B(바탕 모델) + ANIS GeTrans 어댑터(LoRA)를 Transformers 로 돌린다.

바탕 모델 찾기
  1) --base 또는 환경 변수 ANIS_GETRANS_BASE 로 준 폴더(또는 HuggingFace 저장소 이름)
  2) 없으면 model/release.json 의 base_model · base_revision 을 HuggingFace 에서 내려받는다(처음 한 번, 약 9.6GB).
     받은 파일은 HuggingFace 기본 캐시(~/.cache/huggingface, HF_HOME 으로 바꿀 수 있음)에 남아 다음부터는 바로 쓴다.
장치 선택 규칙
  device="auto"  CUDA GPU가 있으면 cuda(bfloat16), 없으면 cpu(bfloat16, 많이 느림)
  device="cuda"  NVIDIA GPU (메모리 약 11GB 필요)
  device="cpu"   CPU (메모리 약 12GB, GPU보다 느림)
번역 방식: 지시문 없이 번역할 글만 넣는다(앞뒤 문맥이 있으면 <prev>/<tgt>/<next> 표지). 탐욕 생성, 출력 최대 256토큰.
ANIS 의 공개 점수(JK Trans 1.3.1)를 잰 것과 같은 입력·생성 방식이다.
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .encoding import DIRECTIONS, InputTooLong, build_source

DEVICES = ("auto", "cpu", "cuda")
EOS, TURN_END, PAD = 1, 106, 0
MAX_PROMPT_TOKENS = 4096


@dataclass
class Item:
    text: str
    prev: list = field(default_factory=list)
    next: list = field(default_factory=list)


@dataclass
class Result:
    text: str
    error: str = ""
    dropped_context: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def default_model_dir():
    """모델 폴더 찾기: $ANIS_GETRANS_MODEL → 실행 파일 옆 model/ → 번들 루트 model/ → ./model"""
    env = os.environ.get("ANIS_GETRANS_MODEL")
    candidates = [Path(env)] if env else []
    if getattr(sys, "frozen", False):
        candidates += [Path(sys.executable).resolve().parent / "model", Path(getattr(sys, "_MEIPASS", "")) / "model"]
    here = Path(__file__).resolve()
    candidates += [here.parent / "model", here.parents[2] / "model", Path.cwd() / "model"]
    for c in candidates:
        if (c / "adapter" / "adapter_model.safetensors").exists():
            return c
    raise FileNotFoundError("모델 폴더를 찾지 못했습니다. --model 또는 ANIS_GETRANS_MODEL 로 지정하세요. "
                            "찾아본 곳: " + ", ".join(str(c) for c in candidates))


def available_devices(model_dir=None):
    out = ["auto", "cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            out.append("cuda")
    except Exception:
        pass
    return out


_LEAD = __import__("re").compile(r"^<tgt>\s*(?:<(?:없음|[A-Z]|대사|시스템|독백|선택지|나레이션)>\s*)*")


def raw_prompt(source):
    """지시문 없는 입력: 문맥 표지가 있으면 표지 붙은 원문 그대로, 없으면 <tgt>·줄 종류 표지를 뗀 맨 문장(줄바꿈은 <br>)."""
    s = source if "<tgt>" in source else "<tgt> " + source.replace("\n", " <br> ")
    if "<prev>" in s or "<next>" in s:
        return s
    return _LEAD.sub("", s).strip()


class Engine:
    def __init__(self, model_dir=None, device="auto", compute_type="auto", threads=0, base=None, max_new=256, **_ignored):
        if device not in DEVICES:
            raise ValueError(f"device는 {', '.join(DEVICES)} 중 하나입니다.")
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        rel = self.model_dir / "release.json"
        self.release = json.loads(rel.read_text(encoding="utf-8")) if rel.exists() else {}
        self.max_new = max_new
        t0 = time.time()
        self._load(device, compute_type, threads, base)
        self.info["load_seconds"] = round(time.time() - t0, 2)
        self.info["version"] = self.release.get("version", "?")

    def _load(self, device, compute_type, threads, base):
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        from peft import PeftModel
        if threads:
            torch.set_num_threads(threads)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU를 찾지 못했습니다. NVIDIA 드라이버와 CUDA판 PyTorch를 확인하거나 --device cpu 를 쓰세요.")
        dtype_name = compute_type if compute_type not in ("auto", None) else "bfloat16"
        if dtype_name not in ("bfloat16", "float16", "float32"):
            raise ValueError("compute_type 은 auto, bfloat16, float16, float32 중 하나입니다.")
        dtype = getattr(torch, dtype_name)
        base = base or os.environ.get("ANIS_GETRANS_BASE")
        revision = None
        if not base:
            base, revision = self.release["base_model"], self.release.get("base_revision")
        kw = dict(revision=revision) if revision else {}
        self.tok = AutoTokenizer.from_pretrained(base, **kw)
        model = AutoModelForImageTextToText.from_pretrained(base, dtype=dtype, **kw)
        model = PeftModel.from_pretrained(model, str(self.model_dir / "adapter")).merge_and_unload()
        self.model = model.to(device).eval()
        self.device, self.dtype = device, dtype
        self.info = dict(backend="transformers", backend_version=transformers.__version__, device=device,
                         compute_type=dtype_name, base_model=str(base), base_revision=revision or "",
                         model=str(self.model_dir / "adapter"))
        if device == "cpu":
            self.info["warning"] = "CPU로 실행합니다. GPU보다 느릴 수 있습니다(NVIDIA GPU 권장)."

    def _ids(self, text):
        s = self.tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def _decode(self, row):
        seq = []
        for t in row:
            if t in (EOS, TURN_END, PAD):
                break
            seq.append(t)
        return self.tok.decode(seq, skip_special_tokens=True).strip()

    def _generate(self, ids, budget=8000, max_batch=16, progress=None):
        import torch
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        out, o = [None] * len(ids), 0
        ctx = torch.autocast("cuda", dtype=self.dtype) if self.device == "cuda" else torch.autocast("cpu", enabled=False)
        with torch.no_grad():
            while o < len(order):
                width = len(ids[order[o]])
                n = max(1, min(max_batch, budget // (width + self.max_new)))
                chunk = order[o:o + n]; o += len(chunk)
                x = torch.full((len(chunk), width), PAD, dtype=torch.long)
                for j, i in enumerate(chunk):
                    x[j, width - len(ids[i]):] = torch.tensor(ids[i])   # 왼쪽 채움
                x = x.to(self.device)
                with ctx:
                    y = self.model.generate(input_ids=x, attention_mask=x.ne(PAD), do_sample=False, max_new_tokens=self.max_new,
                                            eos_token_id=[EOS, TURN_END], pad_token_id=PAD)
                for j, i in enumerate(chunk):
                    out[i] = self._decode(y[j, width:].tolist())
                if progress:
                    progress(o, len(order))
            # 여러 문장을 한꺼번에 생성하면 가끔 한 묶음이 통째로 빈 결과가 된다 → 빈 결과만 한 문장씩 다시 생성
            for i in [i for i in range(len(ids)) if not out[i]]:
                x = torch.tensor([ids[i]], device=self.device)
                with ctx:
                    y = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=self.max_new,
                                            eos_token_id=[EOS, TURN_END], pad_token_id=PAD)
                out[i] = self._decode(y[0, len(ids[i]):].tolist())
        return out

    # ---------- 번역 ----------
    def translate(self, items, direction="ja-ko", line_type="대사", batch_size=16, progress=None):
        """items: Item 또는 문자열 목록. Result 목록을 같은 순서로 돌려준다.
        방향은 모델이 원문 언어로 정한다(가나 → 일→한, 한글 → 한→일). direction 은 형식 확인에만 쓴다."""
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        items = [Item(x) if isinstance(x, str) else x for x in items]
        results, todo, ids = [None] * len(items), [], []
        for i, it in enumerate(items):
            if not it.text.strip():
                results[i] = Result("")
                continue
            src = build_source(it.text, it.prev, it.next, line_type) if (it.prev or it.next or "<tgt>" in it.text) else it.text
            p = self._ids(raw_prompt(src))
            if len(p) > MAX_PROMPT_TOKENS:
                results[i] = Result("", error=f"입력이 너무 깁니다 ({len(p)} 토큰, 최대 {MAX_PROMPT_TOKENS}). 문맥 줄 수를 줄여 주세요.")
                continue
            todo.append(i); ids.append(p)
        if ids:
            outs = self._generate(ids, max_batch=max(1, batch_size), progress=progress)
            for i, p, o in zip(todo, ids, outs):
                src = items[i].text
                if "\n" in src and "<br>" not in src:
                    o = o.replace(" <br> ", "\n").replace("<br>", "\n")
                results[i] = Result(o, input_tokens=len(p), output_tokens=len(self.tok(o, add_special_tokens=False)["input_ids"]))
        return results

    def translate_text(self, text, direction="ja-ko", prev=(), next=(), line_type="대사"):
        r = self.translate([Item(text, list(prev), list(next))], direction, line_type)[0]
        if r.error:
            raise InputTooLong(r.error)
        return r.text
