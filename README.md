# 실습: 나만의 MCP 서버, 그리고 의료 Multi-Agent와 Distillation

맥에서 무료로, API 키 없이 Jupyter Notebook으로 차례로 만들어 봅니다.

- **1일차 (노트북 1-1 \~ 1-6)**: Embedding, Vector DB, RAG, MCP, Agent. 직접 만든 MCP 서버를 Kiro나 Claude에 연결
- **2일차 발표자료**: [`slides/multi-agent-distill.html`](slides/multi-agent-distill.html) (내려받아 브라우저로 열기. ← → 이동, N 발표자 노트)
- **2일차 (노트북 2-1 \~ 2-6)**: 신약 후보물질을 검토하는 Multi-Agent(전문가 3명 + 지휘자)를 만들고, 지휘자의 판단을 작은 모델에 Distillation(증류)으로 가르치기. 이어서 내 주제로 팀을 만들어 웹 화면과 Kiro에서 쓰기(완성 예시: 흉부 엑스레이 판독 보조 팀). [바로 가기](#2일차-의료-multi-agent와-distillation)

> **2일차 수업 전에**: [수업 전에 할 일](#2일차-수업-전에-할-일)을 먼저 보세요. 명령 세 줄이고, 내려받는 데 시간이 걸리니 미리 해 두면 좋습니다.

| 서버 | 하는 일 | Tool(도구) |
|---|---|---|
| `memory_mcp.py` | "이거 기억해" 하면 저장하고, 나중에 의미로 찾거나 한 달치를 꺼내 요약 | `remember`, `recall`, `list_memories`, `forget` |
| `image_mcp.py` | 사진(예: 엑스레이)을 등록하면 설명문을 만들어 Vector(벡터)로 저장하고, 말로 검색 | `add_image`, `add_folder`, `search_images`, `show_image` |

```
[Claude 등 MCP Client] ──MCP──▶ [내 MCP 서버] ──▶ Ollama (Embedding, Vision 모델)
                                      └──▶ Chroma (Vector DB, 내 컴퓨터 폴더에 저장)
```

## 발표 자료

강의 슬라이드는 [`slides/chatgpt-4years.html`](slides/chatgpt-4years.html)에 있습니다. 저장소를 받은 뒤 파일을 더블클릭해 브라우저로 열고, 화살표 키(← →)로 넘깁니다. `F`를 누르면 전체 화면입니다.

## 빠른 시작: 그대로 복사해 붙여 넣기

실습은 **Jupyter Notebook(브라우저)** 으로 합니다. 터미널(Spotlight에서 "터미널" 검색)을 열고, 회색 상자를 통째로 복사해 붙여 넣은 뒤 Enter를 누르면 됩니다.

**1단계. Ollama 앱 설치** (브라우저에서 한 번만)

https://ollama.com/download 에서 내려받아 설치하고, 앱을 한 번 실행합니다. 메뉴 막대에 라마 아이콘이 보이면 됩니다.

**2단계. 실습 코드 받기와 준비**

```bash
git clone https://github.com/gnidoc327/skku-kdt-mcp-lab.git mcp-lab
cd mcp-lab
bash setup.sh
```

모델을 합쳐서 약 16GB 내려받기 때문에 Wi-Fi에 따라 20\~60분 걸립니다(1일차에 받은 모델은 다시 받지 않습니다). **수업 첫 시간에 먼저 실행해 두고** 강의를 들으면 실습 시간에 바로 시작할 수 있습니다. 중간에 끊겨도 다시 실행하면 이어서 받습니다.

`✅ 성공`이 나온 뒤 브라우저에 JupyterLab이 열리면 준비 끝입니다. 다음에 다시 열 때도 `cd mcp-lab` 후 `bash setup.sh`를 실행하면 됩니다.

**Python은 따로 설치하지 않아도 됩니다**

맥에 기본으로 들어 있는 `python3`은 3.9.6입니다(`/usr/bin/python3`, Apple이 넣어 둔 것이라 업그레이드할 수 없고, 3.9는 2025년 10월에 지원이 끝났습니다). 실습 패키지는 3.10 이상이 필요해서, `setup.sh`가 **이 폴더 안에만** Python 3.13을 받아 씁니다.

| | 위치 | 바뀌는 것 |
|---|---|---|
| 맥의 기본 Python 3.9 | `/usr/bin/python3` | 그대로 둡니다 |
| 실습 전용 Python 3.13 | `mcp-lab/.tools/` (약 100MB) | 새로 받습니다. 관리자 암호 필요 없음 |
| 실습 패키지 | `mcp-lab/.venv/` | 여기에만 설치합니다 |
| `PATH`, `~/.zshrc` | | **건드리지 않습니다** |

- 터미널에서 `python3 --version`을 치면 실습 뒤에도 여전히 3.9.6이 나옵니다. 정상입니다.
- 그래서 실습 명령은 항상 **`.venv/bin/python`** 으로 시작합니다. 그냥 `python3 ...`이나 `pip install ...`을 치면 맥의 3.9로 실행되어 `SyntaxError`나 `No matching distribution` 오류가 납니다.
- JupyterLab 안에서는 신경 쓸 필요가 없습니다. 노트북은 자동으로 `.venv`의 3.13으로 돕니다.
- Homebrew나 python.org로 Python을 이미 설치했어도 충돌하지 않습니다. 실습은 폴더 안의 것만 씁니다.
- 지울 때는 `mcp-lab` 폴더를 휴지통에 넣으면 됩니다. 맥의 다른 곳에는 남는 것이 없습니다(Ollama 모델은 `ollama rm 모델이름`으로 따로 지웁니다).

<details>
<summary><b>setup.sh가 하는 일을 직접 하려면</b></summary>

```bash
python3 --version                     # 3.10 이상인지 확인. 3.9면 아래 명령은 실패합니다
python3 -m venv .venv                 # 가상환경 만들기
.venv/bin/python -m pip install jupyterlab ipykernel
.venv/bin/python -m jupyter lab --notebook-dir=example
```

`source .venv/bin/activate`로 가상환경을 켜면 그 터미널 창에서만 `python`이 `.venv`의 것으로 바뀝니다. 창을 닫으면 원래대로 돌아갑니다. 각 노트북의 첫 셀이 필요한 패키지를 설치합니다.
</details>

**3단계. 노트북을 순서대로 실행**

왼쪽 파일 목록에서 `1-1_check_setup`부터 번호 순서대로 엽니다. 셀을 하나씩 선택하고 **Shift + Enter**를 누르면 실행됩니다. `# TODO`가 적힌 곳은 내용을 바꿔서 다시 실행해 보세요.

| 순서 | 노트북 | 내용 |
|---|---|---|
| **1-1** | `1-1_check_setup.ipynb` | Ollama 연결과 모델 확인(없으면 내려받기), 설정 저장 |
| **1-2** | `1-2_embedding_similarity.ipynb` | Embedding으로 뜻이 비슷한 글 찾기, 키워드 검색과 비교, 의미 지도 그리기 |
| **1-3** | `1-3_vector_db.ipynb` | Vector DB(Chroma)에 저장하고 의미와 꼬리표로 찾기 |
| **1-4** | `1-4_rag.ipynb` | RAG: 문서를 찾아서 LLM에게 함께 보여 주기. RAG 없이 물었을 때와 비교 |
| **1-5** | `1-5_mcp_server.ipynb` | Memory MCP 서버에 연결해 도구 목록을 받고 직접 호출 |
| **1-6** | `1-6_agent_with_mcp.ipynb` | Agent: 로컬 LLM이 스스로 MCP 도구를 고르게 하기 (`for`문 하나) |

전부 내 맥북 안에서 돌아가므로 API 키도 요금도 없습니다.

**4단계. (선택) 같은 MCP 서버를 Kiro나 Claude에 연결**

노트북에서 쓴 `memory_mcp.py`를 그대로 AI 도구에 붙입니다. 노트북에서 저장한 기억을 그 AI가 꺼냅니다. 쓰는 도구 하나만 고르면 됩니다.

Kiro:

```bash
.venv/bin/python connect_kiro.py
```

Kiro의 사용자 설정(`~/.kiro/settings/mcp.json`)에 추가됩니다. Kiro는 설정 파일이 바뀌면 자동으로 다시 연결합니다. 안 보이면 명령 팔레트(⌘⇧P)에서 "MCP"를 검색해 설정을 열어 보세요.

Claude Desktop 앱:

```bash
.venv/bin/python connect_claude_desktop.py
```

실행한 뒤 Claude Desktop을 완전히 종료(⌘Q)했다가 다시 켭니다.

Claude Code:

```bash
claude mcp add memory -- "$(pwd)/.venv/bin/python" "$(pwd)/memory_mcp.py"
```

세 경우 모두 기존 설정은 `.bak` 파일로 백업됩니다(Claude Code 제외). 연결한 뒤 채팅창에 그대로 입력해 보세요.

```
이거 기억해 줘: 다음 주 화요일까지 RAG 과제 제출
```

```
과제 마감 언제였지?
```

**안 될 때**

| 증상 | 해결 |
|---|---|
| `Ollama 앱이 꺼져 있습니다` | 응용 프로그램에서 Ollama를 실행하고 `bash setup.sh` 다시 |
| `command not found: git` | 뜨는 설치 창에서 "설치"를 누르고, 끝나면 2단계 다시 |
| `실습용 Python을 내려받지 못했고` | 인터넷(github.com 접속)을 확인하고 `bash setup.sh` 다시. 계속 안 되면 https://www.python.org/downloads/ 에서 Python 3.13 설치 후 다시 |
| `SyntaxError`, `EOL while scanning`, `No matching distribution` | 맥의 기본 Python 3.9로 실행된 것입니다. `python3 ...` 대신 `.venv/bin/python ...`으로 실행하거나 `bash setup.sh`를 다시 실행 |
| 노트북 셀이 `[*]`에서 오래 멈춤 | 모델을 처음 부르는 중입니다. 1\~2분 기다리면 됩니다 |
| `setup.sh`가 중간에 멈춤 | 같은 폴더의 `setup.log` 파일과 화면의 메시지를 강사에게 보여 주세요. 폴더를 옮겼거나 예전 `.venv`가 남아 있으면 `rm -rf .venv` 후 `bash setup.sh` 다시 |
| `config.json이 없습니다` | `1-1_check_setup` 노트북을 먼저 끝까지 실행 |
| Claude에 도구가 안 보임 | Claude Desktop을 ⌘Q로 완전히 끄고 다시 켜기 |
| Kiro에 도구가 안 보임 | 명령 팔레트(⌘⇧P)에서 "MCP" 검색 → 사용자 MCP 설정 열기. `memory`가 있는지 확인하고 저장 |

## 2일차: 의료 Multi-Agent와 Distillation

AWS 기술 블로그의 [SK바이오팜 해커톤 사례](https://aws.amazon.com/ko/blogs/tech/agentic-ai-drug-development-workflow-automation-challenge/)와 같은 구조를 내 맥 안에서 만듭니다. 약 이름을 넣으면 전문가 Agent 세 명이 공개 데이터베이스를 조회해 분석하고, 지휘자가 통합 리포트를 씁니다.

```
질문 ─▶ 지휘자(Orchestrator) ─┬─▶ 의약화학 전문가   (PubChem, RDKit)
                              ├─▶ 임상약리 전문가   (PubMed, openFDA)
                              └─▶ 임상·규제 전문가  (ClinicalTrials.gov)
                                        │
                              지휘자가 세 답을 모아 통합 리포트
```

| 블로그의 구성 | 이 실습 |
|---|---|
| Amazon Bedrock의 Claude | 내 맥의 Ollama `gemma4:e4b` (0원) |
| Strands Agents | Strands Agents (같음) |
| SageMaker의 TxGemma로 분자 특성 예측 | RDKit으로 계산 (GPU 필요 없음) |
| PubChem, PubMed, ClinicalTrials.gov, OpenFDA | 같음 (`drug_tools.py`, API 키 필요 없음) |

### 2일차 수업 전에 할 일

내려받을 것이 약 10GB라서 Wi-Fi에 따라 15\~40분 걸립니다. **수업 전날이나 쉬는 시간에 미리** 해 두세요. 1일차에 받은 것은 다시 받지 않습니다.

**1) Ollama 앱을 켭니다.** 메뉴 막대에 라마 아이콘이 보이면 됩니다.

**2) 터미널에 아래 네 줄을 붙여 넣습니다.** (`mcp-lab` 폴더를 다른 곳에 받았다면 첫 줄을 그 위치로 바꿉니다. 1일차에 폴더를 만들지 않았다면 위의 [빠른 시작](#빠른-시작-그대로-복사해-붙여-넣기)의 1\~2단계를 대신 합니다)

```bash
cd ~/mcp-lab
git checkout .
git pull
bash setup.sh
```

노트북 파일 이름이 1일차는 `1-1` \~ `1-6`, 2일차는 `2-1` \~ `2-6`으로 바뀌었습니다. `git checkout .`은 1일차에 노트북을 실행하며 남은 출력을 지워서 새 이름으로 받을 수 있게 합니다(이 줄을 빼면 `git pull`이 멈춥니다). 저장해 둔 기억(`example/.lab-data`)과 `my_team.py`는 지워지지 않습니다.

`setup.sh`가 알아서 하는 일입니다. 중간에 끊겨도 다시 실행하면 이어서 합니다.

| 단계 | 하는 일 | 새로 받는 것 |
|---|---|---|
| 1/6 | Ollama가 켜져 있는지 확인 | |
| 2/6 | 실습 전용 Python 준비 (맥의 Python은 그대로) | |
| 3/6 | 2일차 패키지 설치 (Strands Agents, RDKit, MLX) | 약 0.5GB, 2\~3분 |
| 4/6 | Ollama 모델 내려받기 | `gemma4:e4b` 9.5GB (가장 오래 걸림) |
| 5/6 | 증류할 작은 모델 내려받기 | `gemma-3-1b` 0.7GB |
| 6/6 | 동작 확인: 큰 모델이 사진을 보는지, 사례 검색이 되는지, 작은 모델이 도는지 | 1\~2분 |

**3) `6/6 동작 확인` 아래에 `❌` 없이 `✅`만 나오고 JupyterLab이 열리면 준비 끝입니다.** 터미널에서 `Ctrl+C`를 두 번 눌러 닫아 두었다가, 수업 때 `bash setup.sh`로 다시 엽니다(두 번째부터는 1\~2분).

**4) `❌`가 나오면** 화면의 안내를 따르고, 해결이 안 되면 `mcp-lab` 폴더의 `setup.log` 파일을 강사에게 보여 주세요.

미리 확인할 것:

- 디스크 여유 공간 20GB 이상 (Apple 메뉴 → 이 Mac에 관하여 → 추가 정보 → 저장 공간)
- 수업 중에는 메모리를 많이 쓰는 앱(Chrome 탭 여러 개, Docker 등)을 닫아 둡니다. 큰 모델이 메모리를 10GB쯤 씁니다
- Kiro로도 해 보려면 Kiro에 로그인해 둡니다

**노트북 순서** (수업에서는 2-1 \~ 2-4를 하고, 2-5와 2-6은 심화 과제입니다)

| 순서 | 노트북 | 내용 | 실행 시간 |
|---|---|---|---|
| **2-1** | `2-1_drug_tools_agent.ipynb` | 공개 데이터베이스 조회 도구 6개와, 그 도구를 쓰는 전문가 Agent 한 명 | 1분 |
| **2-2** | `2-2_multi_agent.ipynb` | 전문가 세 명 + 지휘자. 종합 분석(전원)과 스마트 컨설팅(필요한 전문가만), 리포트 저장 | 3\~4분 |
| **2-3** | `2-3_my_multi_agent.ipynb` | **내 팀 만들기.** 완성 예시(흉부 엑스레이 판독 보조 팀)를 돌려 보고, 건강검진 상담 팀 예시를 고쳐 내 주제의 팀을 만든 뒤 웹 화면과 Kiro로 띄우기 | 4\~5분 |
| **2-4** | `2-4_distill_my_team.ipynb` | 2-3 팀의 "전문가 고르기"를 작은 모델에 증류하고 서비스에 넣기 | 8\~9분 |
| **2-5** | `2-5_distill_data.ipynb` | (심화) 증류를 한 단계씩: 스승 모델(`gemma4:e4b`)이 질문 약 500개에 "누구에게 보낼지" 답을 달아 학습 자료 만들기 | 2\~3분 |
| **2-6** | `2-6_distill_train.ipynb` | (심화) 제자 모델(`gemma-3-1b`, 스승의 10분의 1 이하)을 MLX LoRA로 학습, 학습 전후 정답률과 속도 비교, 제자를 지휘자 자리에 넣어 실행 | 7\~8분 |

신약 팀 증류(2-5, 2-6)를 발표자 맥(M1 Max 32GB)에서 돌린 결과: 학습 약 5분, 스승과 같은 답을 고른 비율이 학습 때 본 유형의 질문에서 **22% → 88%**, 처음 보는 유형에서 **0% → 71%**, Adapter 파일 8MB. 세 번 돌려 처음 보는 유형은 68\~71%였고, 숫자는 실행할 때마다 조금씩 달라집니다.

### 내 Multi-Agent 서비스 만들기 (노트북 2-3, 2-4)

노트북 2-1, 2-2가 따라 만들기였다면, 여기서는 **전문가 목록만 적으면 도는 틀**(`agent_team.py`)로 내 주제의 팀을 만듭니다.

```
                              ┌─▶ 영상 판독 보조 전문가  (사진 다시 보기: Vision 모델)
사진 + 질문 ─▶ 지휘자 ────────┼─▶ 유사 사례 전문가      (사람이 판독 요약을 붙인 사례 검색)
        ▲                     └─▶ 문헌 전문가           (PubMed)
        └─ "누구에게 보낼지"는 증류한 작은 모델이 고름 (노트북 2-4)
```

| 파일 | 내용 |
|---|---|
| `agent_team.py` | 틀. 지휘자(고르기, 동시에 시키기, 합치기)와 라우팅 증류가 들어 있음 |
| `lab_ui.py` | 오래 걸리는 일의 진행 상황(도구 호출, 경과 시간)과 답을 화면에 보여 주는 도우미 |
| `xray_team.py` | 완성 예시: 흉부 엑스레이 판독 보조 팀. 도구 3개와 전문가 3명 |
| `my_team.py` | 내 팀. 노트북 2-3의 셀이 만들어 줌 (예시: 건강검진 결과 상담 팀) |
| `team_service.py` | 팀을 웹 화면이나 MCP 서버로 띄움 |
| `data/xray/` | 공개 라이선스 흉부 엑스레이 9장과 사람이 쓴 판독 요약. 출처는 `SOURCES.md` |

**웹 화면으로 쓰기**

노트북 2-3의 "5. 서비스로 띄우기" 셀을 실행하면 됩니다(`start_service("xray_team")`). 터미널에서 띄우려면 실습 폴더에서 아래를 실행합니다.

```bash
.venv/bin/python team_service.py xray_team
```

브라우저가 저절로 열립니다(http://127.0.0.1:8000). 8000번을 다른 프로그램이 쓰고 있으면 8001번처럼 다음 번호로 열리고, 주소는 터미널에 나옵니다. 사진과 질문을 고르면 어떤 전문가가 불렸는지, 각자 어떤 도구를 썼는지, 통합 답변이 나옵니다. 내 팀은 `xray_team` 자리에 `my_team`을 넣습니다. 내 컴퓨터 안에서만 열립니다.

**Kiro에서 쓰기**

```bash
.venv/bin/python connect_kiro.py --team xray_team
```

Kiro 채팅창에 `lobar_pneumonia.jpg 사진을 xray 팀에게 종합 검토시켜 줘`라고 입력합니다. 이때는 Kiro가 지휘자가 아니라 창구입니다. `ask_team` 도구 하나만 부르고, 전문가를 고르고 분석하는 일은 내 컴퓨터의 팀이 합니다.

**알아 둘 점**

- 증류한 것은 **라우팅(누구에게 보낼지)뿐**입니다. 사진을 보는 일은 큰 모델(`gemma4:e4b`)이 그대로 합니다. 사진을 보는 모델을 증류하려면 믿을 수 있는 정답이 먼저 있어야 하는데, 지금의 스승 모델은 사진을 틀리게 읽는 일이 있어 그대로 가르치면 틀린 답을 배웁니다.
- 1일차 `image_mcp.py`는 Vision 모델이 쓴 설명문으로 사진을 검색했습니다. 이번 사례 검색은 **사람이 쓴 판독 요약**을 근거로 씁니다.
- 실제 환자 사진은 넣지 마세요. `data/xray`의 사진은 모두 Wikimedia Commons의 공개 라이선스 자료입니다.
- 사진 속 소견은 자주 틀립니다. 발표자 맥에서 늑골 골절 사진에 "갈비뼈 선이 끊긴 곳이 있는지 봐 줘"라고 물었더니 "명확하게 끊어진 부분은 보이지 않는다"고 답했습니다. 그래서 리포트의 마지막 항목이 "사람이 꼭 확인할 것"입니다.

발표자 맥(M1 Max 32GB)에서 돌린 결과: 엑스레이 팀 종합 검토 약 80초(전문가 3명), 전문가 한 명만 부르는 질문은 25\~40초. 라우팅 증류는 질문 약 260개에 스승이 답을 다는 데 1분, 학습 5분 30초. 스승과 같은 답을 고른 비율이 학습 때 본 유형에서 **0% → 90\~95%**, 처음 보는 유형에서 **0% → 86\~98%**(네 번 실행). 전문가를 고르는 시간은 스승 0.1\~0.3초, 제자 약 0.2초로 비슷했습니다. 이 맥에서 증류로 얻는 것은 속도가 아니라, 10분의 1 크기의 모델로도 같은 판단을 한다는 점입니다. "종합적으로 검토해 줘"처럼 낱말 자리가 없는 질문은 한 번만 나와 제자가 못 배우던 문제가 있어, 이런 질문은 항상 학습에 세 번씩 넣도록 고쳤습니다. 그래도 "lobar_pneumonia.jpg 사진을 종합 검토해 줘"처럼 파일 이름이 앞에 붙은 질문에는 전문가를 한 명만 골랐습니다. 이 "처음 보는 유형"은 학습한 질문 틀과 표현이 가까운 편이고, 말투가 아주 다른 질문 세 개에서는 두 개만 스승과 같았습니다. 건강검진 예시 팀(`my_team`)은 실행만 확인했고 증류는 돌려 보지 않았습니다.

### Kiro로 이어서 개발하기

노트북에서 만든 팀을 Kiro에게 말로 시켜서 계속 고칠 수 있습니다. 준비는 한 줄입니다.

```bash
cd ~/mcp-lab
.venv/bin/python connect_kiro.py --dev
```

그다음 Kiro에서 **File → Open Folder**로 `mcp-lab` 폴더를 열고(폴더를 믿느냐고 물으면 믿는다(Trust) 쪽을 고릅니다), 채팅창에 입력합니다.

```
xray_team.py를 본보기로 my_team.py에 '복약 상담 팀'을 만들어 줘. 전문가는 두 명, 만든 뒤 질문 하나로 돌려서 확인해 줘
```

이 한 줄이 해 주는 일은 두 가지입니다.

- **"실행해도 될까요?"를 덜 묻습니다.** Kiro는 원래 명령을 실행하거나 파일을 고칠 때마다 허락을 구합니다. `--dev`는 실습에 늘 쓰는 안전한 일만 미리 허락해 둡니다(`~/.kiro/settings/permissions.yaml`).
- **Kiro가 이 폴더를 압니다.** `.kiro/steering/mcp-lab.md`에 실행 방법(`.venv/bin/python`), 파일 구성, 팀 파일 쓰는 법, 의료 자료를 다룰 때 지킬 것을 적어 두었습니다. Kiro는 폴더를 열면 이 문서를 먼저 읽습니다. 내 규칙을 더하고 싶으면 이 파일을 고치세요.

| 묻지 않고 하는 일 | 여전히 매번 묻는 일 |
|---|---|
| 가상환경 파이썬으로 코드 실행 (`.venv/bin/python …`) | 파일 삭제 (`rm`) |
| 파일 보기 (`ls`, `cat`, `grep` 등), `git status`·`diff`·`commit` | `git push` |
| `mcp-lab` 폴더 안의 파일 고치기 | 패키지 설치 (`pip install`), 모델 받기 (`ollama pull`) |
| 실습 MCP 도구 (`memory`, `images`, `drug-review`, `team-*`) | 증류 학습 (5분 넘게 걸리고 컴퓨터가 뜨거워짐) |
| 내 컴퓨터 안의 웹 화면 확인 (`curl http://localhost…`) | 여기에 없는 그 밖의 모든 명령 |

- 되돌리려면 `.venv/bin/python connect_kiro.py --dev-off`. 넣었던 규칙만 빠지고, 원래 있던 규칙은 그대로입니다.
- **전부 허용하기**: 그래도 Kiro가 자꾸 물어 실습이 끊기면 `.venv/bin/python connect_kiro.py --dev-all`(노트북 2-3에서는 `connect_kiro("--dev-all")`)을 씁니다. Kiro가 아무것도 묻지 않고 전부 합니다(명령 실행, 파일 고치기와 지우기, 패키지 설치, 증류 학습, 모든 MCP 도구). **이 폴더뿐 아니라 Kiro로 여는 모든 폴더에 적용되므로 실습 때만 켜고, 끝나면 `--dev-off`로 되돌리세요.** `sudo`와 홈 폴더나 컴퓨터 전체를 지우는 명령은 막아 두었고, `.git`과 Kiro 설정 파일은 Kiro가 여전히 묻거나 막습니다. Kiro CLI 3.5.0에서 파일 지우기, 폴더 밖에 쓰기, `pip` 실행이 묻지 않고 되고 `sudo`가 막히는 것을 확인했습니다.
- 이 규칙은 내 컴퓨터의 사용자 설정이라 다른 폴더를 열어도 적용됩니다. 파일 고치기는 `mcp-lab` 폴더 안으로만 묶었지만, 파일 보기와 `.venv/bin/python` 실행은 다른 폴더에서도 묻지 않습니다. 또 허용한 명령 뒤에 `> 파일`을 붙이면 폴더 밖에도 쓸 수 있고, 이것은 규칙으로 막히지 않았습니다. 파이썬 실행을 허락한다는 것도 Kiro가 쓴 코드가 확인 없이 돈다는 뜻입니다. 즉 이 설정은 편의 기능이지 안전장치가 아니니, 중요한 자료가 있는 컴퓨터에서는 실습이 끝나면 `--dev-off`로 꺼 두세요.
- Kiro는 저장소에 든 설정으로는 권한을 풀어 주지 않습니다(내려받은 폴더가 멋대로 권한을 얻지 못하게 하려는 것). 그래서 각자 이 명령을 한 번 실행해야 합니다.
- Kiro CLI 3.5.0(IDE와 같은 설정 파일을 씁니다)으로 확인한 결과: 위 표의 왼쪽은 묻지 않고 실행됐고, 오른쪽(`rm`, `pip install`, `mlx_lm lora`, `git push`, `ollama pull`, 파일 쓰기 도구로 폴더 밖에 쓰기, 기억 삭제)은 모두 허락을 기다렸습니다. Kiro가 `echo "---"`를 섞어 쓴 명령은 `echo`를 허용하지 않아 물었습니다. `--dev-off` 뒤에는 파이썬 실행과 파일 고치기도 다시 물었습니다.
- 더 넓게 풀고 싶으면 Kiro 명령 팔레트(⌘⇧P)의 "Kiro: Approve For Me"를 쓸 수 있습니다. 이 실습에서는 권하지 않습니다.
- Kiro 요금(2026년 11월 기준, [kiro.dev/pricing](https://kiro.dev/pricing/)): 무료 요금제는 매달 50 credit입니다. 넉넉한 양이 아니니, 한 번에 할 일을 분명하게 적어 시키세요. 학교가 대상 목록에 있으면 [학생 혜택](https://kiro.dev/students/)으로 1년 동안 매달 1,000 credit을 받을 수 있습니다.

**Kiro에서 한 번에 하기 (신약 검토)**

같은 도구를 MCP 서버(`drug_tools_mcp.py`)로 내놓았습니다. Kiro에 연결하면 Kiro가 지휘자가 되어, 말 한마디로 조회부터 리포트 파일 저장까지 합니다.

```bash
.venv/bin/python connect_kiro.py --drug
```

그다음 **Kiro에서 이 `mcp-lab` 폴더를 열고** 채팅창에 입력합니다. 폴더 안의 `.kiro/steering/drug-review.md`가 검토 절차(전문가 세 명의 역할, 리포트 형식, 지어내지 않기)를 Kiro에게 알려 줍니다. 이 파일을 고치면 전문가를 늘리거나 리포트 형식을 바꿀 수 있습니다.

```
ibuprofen을 다음 개발 단계로 진행해도 되는지 종합 검토해서 리포트로 만들어 줘
```

Claude Code를 쓴다면 `claude mcp add drug-review -- "$(pwd)/.venv/bin/python" "$(pwd)/drug_tools_mcp.py"`로 연결하고, 같은 문서를 읽게 하면 됩니다.

**GPU 서버 버전**: 더 큰 모델을 스승으로 쓰고 싶으면 [`aws/README.md`](aws/README.md)를 보세요. GPU EC2에 Ollama를 띄우고 주소만 바꿉니다(Bedrock 미사용, 시간당 약 1달러).

**주의**: 공개 데이터와 작은 모델로 만든 교육용 실습입니다. 결과를 임상 판단이나 투약 결정에 쓰지 마세요. 같은 조회 결과는 `data/cache`에 저장해 다시 쓰므로, 여러 명이 동시에 실습해도 공개 API 한도에 덜 걸립니다.

아래부터는 자세한 설명과 심화 내용입니다.

## 0. 내 맥에 맞는 모델 고르기

1. **Ollama(올라마) 설치**: https://ollama.com 에서 앱을 받아 설치합니다.
   Docker(도커) 안에 설치하면 맥 GPU를 못 써서 매우 느립니다. 반드시 앱으로 설치하세요.
2. **모델 내려받기** (터미널):
   ```bash
   ollama pull bge-m3      # Embedding(임베딩) 모델, 한국어 지원, 약 1.2GB
   ollama pull gemma3:4b   # 이미지 설명용 Vision(비전) 모델, 약 3.3GB (이미지 실습만)
   ollama pull gemma4:e4b  # (선택) Claude Code·OpenCode를 로컬 모델로 돌릴 때, 약 9.5GB
   ```
   내 맥 메모리는  메뉴 → 이 Mac에 관하여에서 확인하고, 아래 표에서 모델을 고르세요.

   | 통합 메모리 | M3 기종 | 추천 LLM (크기) | Vision | 실습 |
   |---|---|---|---|---|
   | 8GB | iMac, MacBook Air 기본형 | `qwen3:4b` (2.5GB) | `gemma3:4b` 빠듯 | Memory MCP만, 다른 앱 닫기 |
   | 16\~18GB | iMac, Air, MacBook Pro 14 (M3 Pro 18GB) | `qwen3:8b` (5.2GB), `gemma3:12b` (8.1GB) | `gemma3:4b` | 둘 다 가능. 로컬 Agent는 `gemma4:e4b`(Context 16K) |
   | 24GB | iMac·Air·Pro 14 최대 옵션 | `qwen3:14b` (9.3GB), `gpt-oss:20b` (14GB) | `gemma3:12b` | 로컬 Agent `gemma4:e4b` 여유 |
   | 36GB | MacBook Pro M3 Pro·Max | `qwen3:30b` (19GB), `gemma3:27b` (17GB) | `gemma3:27b` | 여유 |
   | 48\~64GB | MacBook Pro M3 Max | `qwen3:32b` (20GB), 64GB면 70B급 (약 43GB) | `qwen3-vl` 큰 모델 | 여유 |
   | 96\~128GB | MacBook Pro M3 Max | `gpt-oss:120b` (65GB) | 자유 | 여유 |

   - 계산법: 모델 크기 + 3GB 정도가 메모리의 70% 안에 들어오면 됩니다. macOS가 통합 메모리의 약 70\~75%만 GPU에 쓰게 하기 때문입니다.
   - 속도는 메모리 대역폭이 좌우합니다. M3 100GB/s, M3 Pro 150GB/s, M3 Max 300\~400GB/s.
   - iMac과 MacBook은 같은 M3 칩이라 표가 같습니다. 팬이 없는 MacBook Air는 오래 돌리면 느려집니다.
   - Embedding 모델 `bge-m3`(1.2GB)는 모든 기종에서 됩니다.

## 1. MCP Inspector로 먼저 테스트

AI 없이 도구가 잘 도는지 브라우저에서 직접 눌러 볼 수 있습니다.

```bash
npx @modelcontextprotocol/inspector .venv/bin/python memory_mcp.py
```

열린 화면에서 Connect → Tools → `remember`에 아무 문장이나 넣고 실행 → `recall`로 찾아보세요.

## 2. Claude에 연결

**Claude Code**

```bash
claude mcp add memory -- /절대경로/mcp-lab/.venv/bin/python /절대경로/mcp-lab/memory_mcp.py
claude mcp add images -- /절대경로/mcp-lab/.venv/bin/python /절대경로/mcp-lab/image_mcp.py
```

**Claude Desktop**: 설정 → 개발자 → 설정 편집에서 `claude_desktop_config.json`에 추가

```json
{
  "mcpServers": {
    "memory": { "command": "/절대경로/mcp-lab/.venv/bin/python", "args": ["/절대경로/mcp-lab/memory_mcp.py"] },
    "images": { "command": "/절대경로/mcp-lab/.venv/bin/python", "args": ["/절대경로/mcp-lab/image_mcp.py"] }
  }
}
```

`/절대경로/mcp-lab`은 터미널에서 `pwd`를 실행해 나온 경로로 바꿉니다. `connect_claude_desktop.py`를 쓰면 자동으로 들어갑니다.
Cursor, Kiro, VS Code 등 MCP를 지원하는 도구도 같은 방식으로 연결합니다.

### 비용이 걱정되면: AI만 바꾸기

MCP 서버는 그대로 두고, 앞에 붙는 AI만 바꿀 수 있습니다.

| 방법 | 비용 | 메모 |
|---|---|---|
| Claude Code + Claude 구독 | 요금제 안에서 추가 비용 없음 | 사용량 한도 있음 |
| Claude Code + Amazon Bedrock | 쓴 만큼. 아래 시연 4단계를 Sonnet 5.5로 하면 약 $0.15 | 회사 AWS 계정, 데이터가 계정 밖으로 안 나감 |
| **Claude Code + Ollama** | 0원, 오프라인 | 수업 기본. 데이터가 내 맥북 밖으로 안 나감. 작은 모델이라 가끔 다시 시켜야 함 |
| OpenCode + Ollama | 0원, 오프라인 | 설정이 더 필요(Code Mode 끄기, AGENTS.md). 아래 설명 참고 |

**Claude Code를 Bedrock으로** (가장 쉬운 길은 `claude` 실행 후 `/setup-bedrock` 마법사)

```bash
export CLAUDE_CODE_USE_BEDROCK=1
export AWS_REGION=us-east-1
export AWS_PROFILE=내-프로필
# 모델을 안 정하면 Opus로 돌아서 비쌉니다. Sonnet으로 고정
export ANTHROPIC_MODEL=us.anthropic.claude-sonnet-5-5
```

- 먼저 Bedrock 콘솔 → Model catalog에서 Anthropic 모델 사용 신청(use case form)을 한 번 합니다.
- 모델 ID 앞의 `us.`는 리전마다 다릅니다(서울 `apac.`, 또는 `global.`). `aws bedrock list-inference-profiles`로 확인하세요.
- 실행 후 `/status`에 `Amazon Bedrock`이 보이면 성공입니다.

**Claude Code + Ollama (로컬, 0원)**

Ollama는 Anthropic 호환 API를 제공해서, Claude Code의 주소만 내 맥북으로 바꾸면 로컬 모델로 돕니다. `claude mcp add`로 연결한 MCP 서버는 그대로 씁니다.

```bash
ollama pull gemma4:e4b    # Google Open Weight, 도구 호출·Vision 지원
# Claude Code 지시문이 길어서 Context를 32K로 늘린 버전을 만든다
printf 'FROM gemma4:e4b\nPARAMETER num_ctx 32768\n' > Modelfile
ollama create gemma4-agent -f Modelfile

export ANTHROPIC_BASE_URL=http://localhost:11434
export ANTHROPIC_AUTH_TOKEN=ollama
export ANTHROPIC_API_KEY=""
claude --model gemma4-agent
```

- 32K Context로 실행하면 약 11GB 메모리를 씁니다. 16GB 맥은 `num_ctx 16384`로 줄이세요.
- 환경변수를 넣은 터미널에서만 로컬 모델로 돌고, 새 터미널은 원래 Claude 계정으로 돌아갑니다.
- 작은 모델은 도구 인자 이름을 자주 틀립니다. MCP 서버가 돌려준 오류를 읽고 스스로 고쳐 다시 부르지만, 가끔 "찾아보겠습니다"라고만 하고 멈추면 한 번 더 시키세요.

**OpenCode + Ollama (로컬, 0원)**

OpenCode 2.0부터 MCP 도구를 코드 실행 도구(`execute`) 안에 감싸서 넘기는 "Code Mode"가 기본입니다. 작은 로컬 모델은 이 구조에서 도구를 못 찾으므로 세 가지를 바꿉니다. 실습 폴더의 `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:11434/v1" },
      "models": { "gemma4-tool": { "name": "Gemma 4 (local)", "tools": true } }
    }
  },
  "mcp": {
    "servers": {
      "memory": { "type": "local", "command": ["/절대경로/mcp-lab/.venv/bin/python", "/절대경로/mcp-lab/memory_mcp.py"], "codemode": false }
    }
  },
  "permissions": [
    { "action": "execute", "resource": "*", "effect": "deny" }
  ]
}
```

같은 폴더의 `AGENTS.md`에 도구 사용 규칙을 적어 줍니다.

```markdown
- "기억해 줘"는 memory_remember, 예전 일을 물으면 memory_recall로 먼저 찾는다.
- "No tool named" 오류가 나면 같은 도구를 한 번 더 호출한다. 답은 한국어로.
```

모델은 온도를 낮춘 버전을 씁니다.

```bash
printf 'FROM gemma4:e4b\nPARAMETER num_ctx 32768\nPARAMETER temperature 0.2\n' > Modelfile
ollama create gemma4-tool -f Modelfile
opencode -m ollama/gemma4-tool
```

- OpenCode를 띄우고 MCP 연결이 끝난 뒤(몇 초) 질문하세요. 띄우자마자 물으면 "도구 없음"이 날 수 있습니다.
- 발표자 맥북에서 네 단계 모두 MCP 도구를 직접 호출했습니다. 다만 Grad-CAM 이미지를 받은 뒤 그 이미지를 새 사진으로 착각하는 일이 있었습니다(덱 "시연 6").

## 3. 이렇게 말해 보세요

**메모리 MCP**
- "이거 기억해 줘: 다음 주 화요일까지 RAG 과제 제출"
- "과제 마감 언제였지?" → `recall`로 찾아 답합니다
- "최근 한 달 기억을 주제별로 요약해 줘" → `list_memories(days=30)`로 꺼내고, 요약은 Claude가 합니다

**이미지 MCP**
- "\~/Downloads/xray 폴더 이미지를 '공개 데이터셋, 흉부'라는 메모로 등록해 줘"
- "폐 아래쪽이 하얗게 보이는 흉부 엑스레이 찾아 줘" → `search_images`
- "첫 번째 결과 보여 줘" → `show_image`로 실제 사진을 Claude가 봅니다

## 실제로 돌려 본 결과 (M1 Max 32GB, Ollama 0.40)

- Memory: "숙제 마감이 언제지?"로 검색하면 "다음 주 화요일까지 RAG 과제 제출"이 1위(distance 0.47). 단어가 달라도 의미로 찾습니다.
- Image: Wikimedia 공개 엑스레이 4장 등록에 약 33초(장당 8초 정도). "폐가 보이는 흉부 엑스레이"로 흉부 사진 2장이 1, 2위.
- 한계: `gemma3:4b`가 정상 흉부 사진에 없는 소견을 쓰거나, 뢴트겐의 손 사진에 엉뚱한 촬영자 이름을 붙였습니다(Hallucination).
  설명이 틀리면 검색도 틀립니다. 그래서 `note`에 사람이 정확한 정보를 적고, 메모리가 넉넉하면 더 큰 Vision 모델을 씁니다.

## 4. 데이터 주의

- 실제 환자 사진, 개인정보가 담긴 사진은 넣지 마세요. 공개 데이터셋을 씁니다.
  - Kaggle "Chest X-Ray Images (Pneumonia)"
  - NIH ChestX-ray14
- 이미지 설명은 일반 Vision 모델이 만든 것이라 의료 판독이 아닙니다. 검색 실습용입니다.
- 저장 위치: `~/.lab-mcp/` (지우면 초기화). 바꾸려면 `MEMORY_DB`, `IMAGE_DB` 환경변수를 씁니다.

## 5. 더 해 보기

오늘 실습은 Embedding, Vector DB, RAG, MCP, 로컬 모델까지 다뤘습니다. 실제 의료 업무에 쓰려면 다음은 평가·관측·안전입니다.

- Evals: 질문과 기대하는 메모를 20쌍쯤 만들어 두고 `recall`이 맞는 메모를 찾는지 매번 자동 측정(Regression Test)
- Observability: Langfuse를 Docker로 띄워 도구 호출을 Tracing
- Safety: 메모리에 "이전 지시를 무시하라" 같은 문장을 넣고 `recall`로 꺼냈을 때 Agent가 따르는지 확인(Prompt Injection)


| 바꿔 보기 | 방법 |
|---|---|
| 다른 Embedding 모델 | `ollama pull qwen3-embedding:0.6b` 후 `EMBED_MODEL=qwen3-embedding:0.6b` (모델을 바꾸면 DB 폴더도 새로) |
| 더 좋은 Vision 모델 | 메모리 16GB 이상이면 `VISION_MODEL=qwen3-vl:8b` |
| 태그로 거르기 | `recall`에 `where={"tags": ...}` 조건 추가 |
| Vector DB 교체 | Chroma 대신 Docker pgvector (덱의 "맥북 실습 세팅" 참고) |

## 6. AWS로 옮기면

| 로컬 | AWS |
|---|---|
| Ollama `bge-m3` | Bedrock Titan Text Embeddings V2 |
| Ollama Vision + 텍스트 임베딩 | Bedrock Nova Multimodal Embeddings (이미지를 바로 벡터로) |
| Chroma | Amazon S3 Vectors, Aurora PostgreSQL(pgvector), OpenSearch |
| 내 맥북에서 실행 | AgentCore Runtime에 배포, AgentCore Gateway로 MCP 노출 |

코드는 `embed()`, `describe()` 함수와 `col` 부분만 바꾸면 됩니다.
