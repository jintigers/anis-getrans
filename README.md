# ANIS GeTrans 0.2.0

일본어 ↔ 한국어 번역 모델입니다. Google의 공개 언어 모델 **Gemma 4 E2B**를 바탕으로 ANIS의 번역 데이터로 다시 학습했습니다(버전 0.2.0).
ANIS Trans보다 무겁고 느리지만, 앞뒤 문맥을 더 잘 읽어 생략된 주어·말투·여러 뜻을 가진 낱말을 더 잘 고릅니다.

- 방향: 일본어 → 한국어, 한국어 → 일본어 (원문 언어로 자동 판별)
- 앞뒤 문맥: 대상 문장 앞뒤의 대사를 같이 넣으면 번역이 더 정확해집니다
- 쓰는 방법 세 가지: 명령줄(CLI), 브라우저 화면(GUI), Ollama·OpenAI 호환 로컬 API 서버 — ANIS Trans와 같은 사용법입니다
- 실행 엔진: PyTorch + Transformers (NVIDIA GPU 권장, 메모리 약 11GB)
- **이 압축 파일에는 ANIS가 학습한 부분(어댑터, 약 185MB)만 들어 있습니다.** 바탕 모델(약 9.6GB)은 처음 실행할 때 HuggingFace에서 자동으로 내려받습니다(인터넷 필요, 한 번만).

## 라이선스 요약 (초안)

- 개인·비영리 이용은 무료입니다.
- 후원·광고·판매·기업 이용 등 **모든 영리 이용에는 별도 상업 라이선스가 필요합니다.**
- **파인튜닝·역변환·재배포는 금지됩니다.**
- 바탕 모델(Gemma 4 E2B와 그 변형)은 각 권리자의 **Apache License 2.0**을 따릅니다. [NOTICE.md](NOTICE.md), [LICENSE-APACHE-2.0.txt](LICENSE-APACHE-2.0.txt)를 보세요.

전문은 [LICENSE-ko.md](LICENSE-ko.md)(정본)와 [LICENSE-en.md](LICENSE-en.md)(영어 번역)에 있습니다. 아직 초안이라 내용이 바뀔 수 있습니다. 프로그램을 처음 실행할 때 이 요약이 한 번 표시됩니다.

---

## 1. 설치

### 필요한 것

| 항목 | 내용 |
|---|---|
| Python | 3.10 이상 (3.10~3.12 권장). Windows는 [python.org](https://www.python.org/downloads/) 설치 때 **"Add python.exe to PATH"** 를 켜 주세요 |
| GPU (권장) | NVIDIA GPU, 메모리 12GB 이상 권장(실제 사용 약 11GB) + 최신 드라이버 |
| 디스크 | 설치 약 6GB(PyTorch 포함) + 바탕 모델 9.6GB |
| 메모리 | GPU로 쓸 때 시스템 메모리 16GB 이상 권장. CPU로 쓰면 약 12GB가 필요하고 PC 성능에 따라 GPU보다 많이 느릴 수 있습니다 |
| 인터넷 | 설치할 때와 처음 번역할 때(바탕 모델 내려받기)만 필요합니다 |

설치 스크립트는 이 폴더 안에 `.venv` 가상환경을 만들고 필요한 패키지(PyTorch, Transformers, PEFT 등)를 내려받습니다. 시스템 Python은 건드리지 않습니다.

### Linux / macOS

```bash
cd anis-getrans-0.2.0
./install.sh            # Linux는 CUDA판 PyTorch가 함께 설치됩니다
./install.sh --cpu      # GPU가 없는 PC (받는 크기가 작음, 번역은 느림)
```

### Windows

폴더에서 `install.cmd` 를 두 번 클릭하거나, 명령 프롬프트에서:

```bat
cd anis-getrans-0.2.0
install.cmd             :: NVIDIA GPU용 CUDA판 PyTorch
install.cmd --cpu       :: GPU가 없는 PC
```

### 처음 실행 — 바탕 모델 내려받기

처음 번역할 때 바탕 모델 `TrevorJS/gemma-4-E2B-it-uncensored`(Gemma 4 E2B 기반, 고정 판 `7a34581e`)을 HuggingFace에서 내려받습니다(약 9.6GB). 받은 파일은 HuggingFace 캐시(기본 `~/.cache/huggingface`, 환경 변수 `HF_HOME` 으로 위치를 바꿀 수 있음)에 남아 다음부터는 바로 시작합니다.

- 인터넷 없이 쓰려면: 한 번 받은 뒤에는 환경 변수 `HF_HUB_OFFLINE=1` 을 켜고 실행하세요.
- 바탕 모델을 이미 다른 폴더에 받아 두었다면 `--base <폴더>` 또는 환경 변수 `ANIS_GETRANS_BASE=<폴더>` 로 지정합니다.

### 설치 확인

```bash
./anis-getrans --info                      # Windows: anis-getrans --info
```

`device: cuda`, `compute_type: bfloat16` 이 보이면 GPU로 실행됩니다.

### 어디서나 `anis-getrans` 로 실행하기 (선택)

- Linux / macOS: `ln -s "$PWD/anis-getrans" ~/.local/bin/anis-getrans` (`anis-getrans-gui`, `anis-getrans-server` 도 같은 방법)
- Windows: 이 폴더를 사용자 환경 변수 `Path` 에 추가하면 어디서나 `anis-getrans` 로 실행됩니다

아래 예시는 모두 `anis-getrans` 로 적었습니다. 이 폴더에서 실행할 때 Linux·macOS는 `./anis-getrans`, Windows PowerShell은 `.\anis-getrans` 로 입력하세요.

---

## 2. 명령줄(CLI) 사용법

```bash
# 단문 번역: 방향은 자동 판별 (가나가 있으면 일→한, 한글이 있으면 한→일)
anis-getrans "今日は雨が降ってるね。"
anis-getrans "오늘 저녁은 뭐 먹을까?"

# 방향 지정 (GeTrans는 원문 언어로 방향을 정하므로, 이 옵션은 일본어·한국어가 섞인 줄의 판별에만 영향을 줍니다)
anis-getrans -d ja-ko "東京"
anis-getrans -d ko-ja "선배, 얼굴색이 안 좋아요."

# 앞뒤 문맥 (--prev 는 오래된 것부터, --next 는 가까운 것부터 여러 번 쓸 수 있음)
anis-getrans --prev "先輩、顔色悪いですよ。" --next "ああ、少し寝不足なだけだ。" "大丈夫？"

# 파일 일괄 번역: 한 줄 = 한 문장, 빈 줄은 그대로
anis-getrans -f examples/sample_ja.txt -o out.txt
anis-getrans -f examples/sample_ja.txt --context-lines 2,1 -o out.txt   # 앞 2줄·뒤 1줄을 문맥으로
anis-getrans -f examples/sample_ja.txt --with-source                     # 원문<TAB>번역

# JSONL: 줄마다 {"text", "prev", "next", "direction"} → "translation" 을 붙여 출력
anis-getrans -f examples/sample.jsonl -o out.jsonl

# 표준입력 파이프
cat examples/sample_ja.txt | anis-getrans -d ja-ko > out.txt

# 인자 없이 실행하면 대화형 모드 (모델을 한 번만 불러옴)
anis-getrans

# 장치 선택
anis-getrans --device cpu  "…"     # CPU (bfloat16, 매우 느림)
anis-getrans --device cuda "…"     # NVIDIA GPU
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `-d`, `--direction` | `auto`, `ja-ko`, `ko-ja` | `auto` |
| `--prev`, `--next` | 앞·뒤 문맥 문장 (여러 번 지정 가능) | 없음 |
| `--context-lines 앞,뒤` | 파일·파이프 번역 때 이웃한 줄을 문맥으로 넣음 | `0,0` |
| `--line-type` | 문맥을 쓸 때 붙이는 줄 종류: 대사, 독백, 나레이션, 선택지, 시스템 | 대사 |
| `-f`, `-o` | 입력 파일 (`.txt`, `.jsonl`), 출력 파일 | 표준입출력 |
| `--format` | `auto`(확장자로 판별), `txt`, `jsonl` | `auto` |
| `--device` | `auto`, `cpu`, `cuda` | `auto` |
| `--compute-type` | `auto`(=bfloat16), `bfloat16`, `float16`, `float32` | `auto` |
| `--threads` | CPU 스레드 수 (0이면 자동) | 0 |
| `--base` | 바탕 모델 폴더나 HuggingFace 저장소 (기본: 자동으로 내려받음) | 자동 |
| `--encoding` | 파일·파이프 문자 인코딩 | `utf-8` |
| `-v`, `--info`, `-V` | 장치·속도 표시 / 장치 정보만 보기 / 버전 | |

번역에 실패한 줄이 있으면 종료 코드가 1이고, 그 줄은 빈 칸으로 남긴 채 이유를 표준오류에 출력합니다.

---

## 3. GUI (브라우저 화면)

```bash
anis-getrans-gui                 # Windows: anis-getrans-gui.cmd 두 번 클릭
anis-getrans --gui               # 같은 동작
```

macOS는 Finder에서 `ANIS GeTrans GUI.command` 를 두 번 클릭해도 됩니다. 실행하면 기본 브라우저에 `http://127.0.0.1:7871/` 이 열립니다. 7871 포트가 사용 중이면 다음 빈 포트를 고르고, 그 주소를 터미널에 표시합니다. 끝낼 때는 터미널 창을 닫거나 Ctrl+C를 누르세요.

- **문장 번역**: 입력창과 출력창, 방향(자동·일→한·한→일, ⇄로 바꾸기), 줄 종류, 앞뒤 문맥 칸이 있습니다. 번역 단축키는 Ctrl+Enter(맥은 ⌘+Enter)입니다. "줄마다 따로 번역"을 켜면 여러 줄을 각각 번역합니다.
- **일괄 번역**: `.txt`/`.jsonl` 파일을 고르거나 붙여넣고, 앞뒤 문맥 줄 수를 정해 번역합니다. 진행률이 표시되고, 결과는 `.txt`(또는 `.jsonl`)나 원문+번역 `.tsv` 로 내려받습니다.
- **장치·정밀도**: 오른쪽 위에서 자동·CPU·GPU(CUDA)를 고르고 "적용"을 누르면 모델을 다시 불러옵니다. 이 컴퓨터에서 쓸 수 없는 장치는 비활성화됩니다.

| 옵션 | 설명 |
|---|---|
| `--port 7871` | 포트 |
| `--device`, `--compute-type` | 처음 불러올 장치·정밀도 |
| `--no-browser` | 브라우저를 자동으로 열지 않음 |
| `--host 127.0.0.1` | 기본은 이 컴퓨터에서만 접속됩니다. `0.0.0.0` 으로 열면 같은 네트워크의 다른 기기도 접속할 수 있으니 주의하세요 |

### 왜 로컬 웹 UI인가

Qt·Tk 데스크톱 앱 대신 **Python 표준 라이브러리 HTTP 서버와 브라우저 화면**을 골랐습니다.

1. **추가 의존성이 없습니다.** Qt(PySide6)는 플랫폼별 바이너리만 100MB가 넘고, Gradio 같은 웹 프레임워크도 의존성이 수십 개입니다. 이 GUI는 CLI와 같은 두 패키지(ctranslate2, sentencepiece)만 씁니다.
2. **일본어·한국어 입력이 안정적입니다.** Tk는 Linux(X11)에서 한글·일본어 입력기(IME) 조합이 자주 깨지고, macOS의 Homebrew·pyenv Python에는 Tk가 빠져 있는 경우가 많습니다. 브라우저는 모든 OS에서 IME·글꼴·복사/붙여넣기를 제대로 처리합니다.
3. **어디서나 같은 화면입니다.** Linux·Windows·macOS에서 똑같이 동작하고, 독립 실행판(PyInstaller)에도 그대로 들어갑니다. 원격 데스크톱이나 SSH 포트 포워딩으로도 쓸 수 있습니다.

서버는 기본으로 127.0.0.1에만 열립니다. 다른 웹사이트가 이 서버로 요청을 보내지 못하도록 전용 헤더 확인도 합니다. 입력한 문장은 컴퓨터 밖으로 나가지 않습니다.

---

## 4. 로컬 API 서버 (Ollama·OpenAI 호환)

Ollama와 vLLM은 이 모델 같은 인코더-디코더 번역 모델을 돌리지 못합니다. 그래서 같은 API 형식으로 응답하는 작은 서버를 넣었습니다. Ollama나 OpenAI API를 쓰는 번역 도구·스크립트에서 주소와 모델 이름만 바꾸면 됩니다.

```bash
anis-getrans-server                          # Windows: anis-getrans-server.cmd
anis-getrans --serve                         # 같은 동작
anis-getrans-server --device cuda --port 11434   # GPU로, Ollama 기본 포트에서 (Ollama를 끈 뒤)
```

기본 주소는 `http://127.0.0.1:11436` 입니다. 실제 Ollama(11434)와 겹치지 않도록 한 칸 옆 포트를 씁니다.

### 엔드포인트

| 형식 | 엔드포인트 |
|---|---|
| Ollama | `GET /api/tags`(모델 목록), `POST /api/generate`, `POST /api/chat`, `POST /api/show`, `GET /api/ps`, `GET /api/version` |
| OpenAI | `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions` |

- 비스트리밍 응답은 각 API의 표준 형식을 따릅니다. OpenAI는 `choices[0].message.content` 와 `usage`, Ollama는 `response` 또는 `message.content` 와 `done`, `eval_count` 등입니다.
- 스트리밍도 됩니다. Ollama는 NDJSON이고 `stream` 을 생략하면 켜집니다(Ollama와 같음). OpenAI는 `"stream": true` 일 때 SSE로 보내고, 마지막에 `data: [DONE]` 을 보냅니다. 번역을 다 마친 뒤 줄 단위로 나눠 보내므로, 글자가 하나씩 나오지는 않습니다.
- `temperature`, `top_p` 같은 샘플링 옵션은 무시합니다. 번역은 항상 빔 탐색 4로 합니다(같은 입력이면 같은 결과).

### 모델 이름과 번역 방향

| 모델 이름 | 방향 |
|---|---|
| `anis-getrans` (`anis-getrans:latest`, `anis-getrans:auto`) | 문장마다 자동 판별 |
| `anis-getrans:ja-ko` | 일본어 → 한국어 |
| `anis-getrans:ko-ja` | 한국어 → 일본어 |

방향은 다음 순서로 정합니다.

1. 요청의 `anis.direction`
2. 모델 이름
3. system 프롬프트 속 방향 표현. 예: "Japanese to Korean", "ko-ja", "일본어로", "한국어로 번역"
4. 문장마다 자동 판별

system 프롬프트의 나머지 지시는 쓰지 않습니다. 이 모델은 지시를 따르는 대화형 모델이 아니라 번역 모델이기 때문입니다.

모델 이름을 바꿀 수 없는 도구(예: `gpt-4o-mini` 로 고정)에는 `--accept-any-model` 을 쓰세요. 모르는 이름도 자동 방향으로 처리합니다.

### 무엇을 번역하나 — 문맥 전달 규칙

- `/api/chat`, `/v1/chat/completions`: **마지막 user 메시지**를 번역합니다.
- `/api/generate`, `/v1/completions`: `prompt` 를 번역합니다.
- 여러 줄을 보내면 기본으로 **줄마다 따로** 번역하고, 줄 구조(빈 줄 포함)를 그대로 돌려줍니다.

문맥은 세 가지 방법으로 넣습니다.

1. **채팅 기록**: 마지막 user 메시지 앞의 user 메시지들(한 줄짜리)이 앞 문맥(`<prev>`)이 됩니다. 기본은 최근 3개이고, assistant 메시지는 쓰지 않습니다. 앞 대사를 차례로 보내는 도구라면 따로 설정하지 않아도 문맥이 들어갑니다.
2. **`anis` 객체**: 요청 본문에 넣습니다. Ollama 요청은 `options` 안에 넣어도 되고, OpenAI 파이썬 SDK는 `extra_body={"anis": {...}}` 로 넣습니다.

   | 키 | 뜻 | 기본값 |
   |---|---|---|
   | `prev` | 앞 문맥 (문자열 또는 목록, 오래된 것부터) | 없음 |
   | `next` | 뒷 문맥 (문자열 또는 목록, 가까운 것부터) | 없음 |
   | `direction` | `auto`, `ja-ko`, `ko-ja` | 모델 이름에 따름 |
   | `line_type` | 대사, 독백, 나레이션, 선택지, 시스템 | 대사 |
   | `history` | 채팅 기록에서 문맥으로 쓸 user 메시지 수 (0이면 안 씀) | 3 |
   | `split_lines` | 여러 줄을 줄마다 번역할지. `false` 면 한 문장으로 보고 줄바꿈을 `<br>` 로 넣음 | `true` |
   | `context_lines` | 줄마다 번역할 때 이웃한 줄을 문맥으로 쓸 수 `"앞,뒤"` | `"0,0"` |

   여러 줄을 줄마다 번역할 때 `prev`·채팅 기록은 첫 줄 앞에, `next` 는 마지막 줄 뒤에 붙습니다.
3. **모델 입력 형식 직접 쓰기**: 메시지에 `<tgt>` 가 들어 있으면 그대로 넣습니다. 예: `<prev> <B> <대사> 準備はいい？ <tgt> <A> <대사> 行くぞ`

### 호출 예

```bash
# Ollama 형식
curl http://127.0.0.1:11436/api/generate -d '{"model":"anis-getrans:ja-ko","prompt":"今日は雨が降ってるね。","stream":false}'
curl http://127.0.0.1:11436/api/chat -d '{"model":"anis-getrans","stream":false,
  "messages":[{"role":"user","content":"大丈夫？"}],
  "options":{"prev":["先輩、顔色悪いですよ。"],"next":"ああ、少し寝不足なだけだ。"}}'

# OpenAI 형식
curl http://127.0.0.1:11436/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"anis-getrans:ko-ja","messages":[{"role":"user","content":"오늘 저녁은 뭐 먹을까?"}]}'
```

파이썬 예제는 [examples/api_openai.py](examples/api_openai.py)(`pip install openai`)와 [examples/api_ollama.py](examples/api_ollama.py)(`pip install ollama`)에 있습니다.

### 서버 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--host` | `127.0.0.1` 이면 이 컴퓨터에서만 접속. `0.0.0.0` 으로 열면 다른 기기도 접속 가능 (`--api-key` 권장) | `127.0.0.1` |
| `--port` | 포트. 사용 중이면 오류를 내고 끝냅니다 (ANIS Trans 서버 11435 와 겹치지 않게 11436) | 11436 |
| `--device`, `--compute-type`, `--threads`, `--base` | 장치·정밀도·CPU 스레드·바탕 모델 | `auto` |
| `--api-key 키` | 지정하면 `Authorization: Bearer 키` 가 있는 요청만 받음 | 없음 |
| `--allow-origin URL` | 브라우저 웹앱에서 호출을 허용할 출처 (여러 번 지정, `*` 는 모두). 기본은 localhost 출처만 허용하고 다른 웹사이트의 브라우저 요청은 막음 | localhost |
| `--accept-any-model` | 모르는 모델 이름도 자동 방향으로 처리 | 끔 |
| `-v` | 요청 기록 출력 | 끔 |

---

## 5. CPU / GPU 선택

| `--device` | 동작 |
|---|---|
| `auto` (기본) | NVIDIA CUDA GPU가 있으면 GPU, 없으면 CPU |
| `cuda` | NVIDIA GPU (메모리 약 11GB) |
| `cpu` | CPU. 메모리 약 12GB가 필요하고 PC 성능에 따라 GPU보다 많이 느릴 수 있습니다 |

- 기본 정밀도는 bfloat16입니다(공개 점수를 잰 것과 같은 설정). `--compute-type float16` 은 bfloat16을 지원하지 않는 오래된 GPU용입니다.
- **macOS / AMD GPU / Intel GPU**: 지금은 CPU로 실행됩니다.
- 번역 품질이 같아도 속도는 ANIS Trans보다 느립니다(GPU에서 한 줄 약 1.2초). 많은 줄을 빨리 옮겨야 하면 ANIS Trans를 쓰세요.

---

## 5-1. GGUF (llama.cpp · Ollama · LM Studio)

설치 없이 llama.cpp 계열 프로그램에서 바로 쓰는 **Q4_K_M GGUF**도 있습니다. 0.2.0 어댑터를 바탕 모델에 합친 뒤 4비트로 줄인 파일입니다(텍스트 전용, 이미지·음성 부분 없음).

| | 배포판(이 프로그램) | GGUF Q4_K_M |
|---|---|---|
| 크기 | 어댑터 185MB + 바탕 모델 9.6GB | 3.4GB 한 파일 |
| 메모리 (문맥 4096) | GPU 약 11GB | GPU 약 2.1GB + RAM 약 2GB, CPU만: RAM 약 3~4GB |
| 번역 방식 | 빔 탐색 4 + 재번역 | 탐욕 생성 (빔·재번역 없음) |
| JK Trans 1.3.1 일→한 / 한→일 | 51.69 / 51.96 | 50.27 / 50.70 |

같은 탐욕 생성으로 잰 원래 정밀도(bf16)는 50.61 / 50.64입니다. 4비트로 줄여서 떨어진 몫은 일→한 −0.34, 한→일 +0.06(차이 없음)입니다.

**받기**
- HuggingFace (한 파일): [jintigers/anis-getrans](https://huggingface.co/jintigers/anis-getrans) 의 `anis-getrans-0.2.0-Q4_K_M.gguf` (라이선스 동의 필요)
  ```bash
  hf download jintigers/anis-getrans anis-getrans-0.2.0-Q4_K_M.gguf --local-dir .
  ```
- GitHub 릴리스 (3조각): [v0.2.0](https://github.com/jintigers/anis-getrans/releases/tag/v0.2.0) 의 `anis-getrans-0.2.0-Q4_K_M-0000N-of-00003.gguf`. GitHub는 파일 하나당 2GB까지라 나눠 올렸습니다. llama.cpp는 첫 조각만 지정하면 나머지를 같이 읽습니다. Ollama·LM Studio에서는 한 파일로 합쳐 쓰세요.
  ```bash
  llama-gguf-split --merge anis-getrans-0.2.0-Q4_K_M-00001-of-00003.gguf anis-getrans-0.2.0-Q4_K_M.gguf
  ```

**Ollama**
```bash
cat > Modelfile <<'MF'
FROM ./anis-getrans-0.2.0-Q4_K_M.gguf
PARAMETER temperature 0
PARAMETER num_predict 256
MF
ollama create anis-getrans -f Modelfile
ollama run --think=false anis-getrans "今日は雨が降ってるね。"
```
API로 부를 때도 `"think": false` 를 넣어 주세요. Ollama는 Gemma 4 모델에 기본으로 생각 모드를 켜는데, 이 모델은 생각 모드 없이 학습했습니다.
```bash
curl http://127.0.0.1:11434/api/generate -d '{"model":"anis-getrans","prompt":"선배, 오늘 같이 갈래요?","think":false,"stream":false}'
```

**llama.cpp**
```bash
llama-server -m anis-getrans-0.2.0-Q4_K_M.gguf -ngl 99 -c 4096
curl http://127.0.0.1:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"おはようございます、先輩！"}],"temperature":0}'
```

- 지시문 없이 번역할 글만 user 메시지로 넣습니다. 방향은 원문 언어로 정해집니다.
- 앞뒤 문맥은 `<prev> 앞 대사 <tgt> 번역할 대사 <next> 뒤 대사` 형식으로 한 메시지에 넣습니다.
- 라이선스는 배포판과 같습니다(개인·비영리 무료, 허가받지 않은 파인튜닝·재배포 제한).

---

## 6. 입력 형식 참고

- **문맥**: 문맥을 주면 모델 입력은 `<prev> <대사> 앞문장 <tgt> <대사> 대상 <next> <대사> 뒷문장` 형태가 됩니다. `<대사>` 자리는 `--line-type` 으로 바꿀 수 있습니다. 문맥이 없으면 `<tgt> 대상` 만 넣습니다.
- **직접 입력**: 문장에 `<tgt>` 가 들어 있으면 이미 모델 형식으로 쓴 것으로 보고 그대로 넣습니다. 화자 태그(`<A>`~`<H>`)도 이때 쓸 수 있습니다. 예: `"<prev> <B> <대사> 準備はいい？ <tgt> <A> <대사> 行くぞ"`
- **줄바꿈**: 한 문장 안의 줄바꿈은 `<br>` 로 바꿔 넣고, 번역문에서 다시 줄바꿈으로 돌립니다. 파일 번역은 한 줄 = 한 문장을 유지하므로 `<br>` 그대로 둡니다.
- **게임 변수**: `<v1>` 같은 자리 표시자는 대체로 그대로 남습니다. 다만 한 문장에 변수가 여러 개면 일부가 빠지거나 바뀔 수 있으니 번역 후 확인하세요.
- **길이**: 문맥 포함 한 줄 입력은 4,096토큰까지입니다. 넘치면 그 줄은 오류로 표시됩니다. 앞뒤 문맥 줄 수를 줄여 주세요.
- **생성**: 빔 탐색 4(같은 입력이면 같은 결과, `--beam 1` 이면 탐욕 생성으로 더 빠름), 한 줄 출력 최대 256토큰(`--max-length`). 빈 출력·번역되지 않고 원문 언어가 남은 출력·같은 말이 되풀이된 출력은 한 줄씩 다시 번역하고, 그래도 안 되면 앞뒤 문맥 없이 대상 줄만 번역합니다. 지시문 없이 번역할 글만 넣는 방식이라 원문 언어로 번역 방향이 정해집니다.

---

## 7. 문제 해결

| 증상 | 해결 |
|---|---|
| 처음 실행이 오래 걸림 | 바탕 모델(약 9.6GB)을 내려받는 중입니다. 한 번만 받습니다 |
| `CUDA out of memory` | GPU 메모리가 부족합니다(약 11GB 필요). 다른 GPU 프로그램을 끄거나 `--device cpu` 로 쓰세요 |
| `CUDA GPU를 찾지 못했습니다` | NVIDIA 드라이버를 확인하세요. Windows는 `install.cmd` 로(CPU판이 아닌) 다시 설치하세요 |
| 인터넷 없이 실행하면 오류 | 바탕 모델을 한 번 받은 뒤 `HF_HUB_OFFLINE=1` 을 켜고 실행하세요 |
| Windows에서 파이프 결과가 깨짐 | 명령 프롬프트: `chcp 65001`. PowerShell 5.1: `$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()` |
| API 서버: `… 에 열 수 없습니다` | 그 포트를 다른 프로그램이 쓰고 있습니다. `--port` 로 다른 포트를 고르세요 |
| API 서버: `model '…' not found` | 모델 이름을 `anis-getrans` 등으로 바꾸거나 `--accept-any-model` 을 쓰세요 |
| GUI가 열리지 않음 | 터미널에 표시된 주소(`http://127.0.0.1:포트/`)를 브라우저에 직접 입력하세요 |

---

## 8. 폴더 구성

```
anis-getrans-0.2.0/
├─ README.md                         이 문서
├─ LICENSE-ko.md, LICENSE-en.md      라이선스 (초안, 한국어 정본·영어 번역)
├─ NOTICE.md, LICENSE-APACHE-2.0.txt 바탕 모델(Gemma 4 E2B) 고지와 Apache License 2.0
├─ install.sh / install.cmd          설치 (Linux·macOS / Windows)
├─ anis-getrans, anis-getrans-gui, anis-getrans-server              실행기 (Linux·macOS)
├─ anis-getrans.cmd, anis-getrans-gui.cmd, anis-getrans-server.cmd  실행기 (Windows)
├─ ANIS GeTrans GUI.command          macOS 두 번 클릭용 GUI 실행기
├─ pyproject.toml                    Python 패키지 정의 (anis-getrans)
├─ src/anis_getrans/                 프로그램 (CLI, 엔진, 입력 인코딩, 웹 GUI, API 서버)
├─ model/
│  ├─ adapter/                       ANIS가 학습한 어댑터 (LoRA)
│  └─ release.json                   버전·바탕 모델 정보
└─ examples/                         예제 입력(txt, jsonl), API 호출 예제
```

## 9. 버전

ANIS GeTrans **0.2.0**은 두 번째 공개판입니다. ANIS Trans와는 따로 개발하며 버전 번호도 따로 셉니다.

- **0.2.0 GGUF** (2026-11-23): 0.2.0 가중치를 합친 Q4_K_M GGUF 추가 (5-1 참고, 탐욕 생성 기준 1.3.1 일→한 50.27, 한→일 50.70).
- **0.2.0** (2026-11-23): 어댑터(가중치)는 0.1.0과 같고 번역 방식이 바뀌었습니다. 빔 탐색 4로 번역하고, 빈 출력·원문 언어가 남은 출력(번역 안 됨)·같은 말이 되풀이된 출력을 다시 번역해 고칩니다. 한 줄 번역은 0.1.0보다 조금 느립니다(`--beam 1` 로 0.1.0과 같은 탐욕 생성을 쓸 수 있습니다).
- **0.1.0** (2026-11-16): 첫 공개판.
