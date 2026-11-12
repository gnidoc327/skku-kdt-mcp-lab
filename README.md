# 실습: 나만의 MCP 서버 만들기

맥북에서 무료로, API 키 없이 MCP(엠씨피) 서버 두 개를 만들어 Claude 같은 AI에 연결합니다.

| 서버 | 하는 일 | Tool(도구) |
|---|---|---|
| `memory_mcp.py` | "이거 기억해" 하면 저장하고, 나중에 의미로 찾거나 한 달치를 꺼내 요약 | `remember`, `recall`, `list_memories`, `forget` |
| `image_mcp.py` | 사진(예: 엑스레이)을 등록하면 설명문을 만들어 Vector(벡터)로 저장하고, 말로 검색 | `add_image`, `add_folder`, `search_images`, `show_image` |

```
[Claude 등 MCP Client] ──MCP──▶ [내 MCP 서버] ──▶ Ollama (Embedding, Vision 모델)
                                      └──▶ Chroma (Vector DB, 내 컴퓨터 폴더에 저장)
```

## 빠른 시작: 그대로 복사해 붙여 넣기

터미널(Spotlight에서 "터미널" 검색)을 열고, 회색 상자를 통째로 복사해 붙여 넣은 뒤 Enter를 누르면 됩니다. 경로를 고칠 곳은 없습니다.

**1단계. Ollama 앱 설치** (브라우저에서 한 번만)

https://ollama.com/download 에서 내려받아 설치하고, 앱을 한 번 실행합니다. 메뉴 막대에 라마 아이콘이 보이면 됩니다.

**2단계. 실습 코드 받기와 준비**

```bash
git clone https://github.com/gnidoc327/skku-kdt-mcp-lab.git mcp-lab
cd mcp-lab
bash setup.sh
```

마지막에 `✅ 성공`이 나오면 준비 끝입니다. 이미지 실습까지 하려면 `bash setup.sh --image`를 한 번 더 실행합니다.

**3단계. Claude에 연결** (둘 중 쓰는 것 하나)

Claude Desktop 앱:

```bash
python3 connect_claude_desktop.py
```

실행한 뒤 Claude Desktop을 완전히 종료(⌘Q)했다가 다시 켭니다. 기존 설정은 `.bak` 파일로 백업됩니다.

Claude Code:

```bash
claude mcp add memory -- uv run "$(pwd)/memory_mcp.py"
```

**4단계. 말 걸기** (Claude 대화창에 그대로 입력)

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
| Claude에 도구가 안 보임 | Claude Desktop을 ⌘Q로 완전히 끄고 다시 켜기 |
| 처음 실행이 1~2분 걸림 | 필요한 패키지를 받는 중입니다. 기다리면 됩니다 |

아래부터는 자세한 설명입니다.

## 0. 준비 (한 번만)

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
   | 16~18GB | iMac, Air, MacBook Pro 14 (M3 Pro 18GB) | `qwen3:8b` (5.2GB), `gemma3:12b` (8.1GB) | `gemma3:4b` | 둘 다 가능. 로컬 Agent는 `gemma4:e4b`(Context 16K) |
   | 24GB | iMac·Air·Pro 14 최대 옵션 | `qwen3:14b` (9.3GB), `gpt-oss:20b` (14GB) | `gemma3:12b` | 로컬 Agent `gemma4:e4b` 여유 |
   | 36GB | MacBook Pro M3 Pro·Max | `qwen3:30b` (19GB), `gemma3:27b` (17GB) | `gemma3:27b` | 여유 |
   | 48~64GB | MacBook Pro M3 Max | `qwen3:32b` (20GB), 64GB면 70B급 (약 43GB) | `qwen3-vl` 큰 모델 | 여유 |
   | 96~128GB | MacBook Pro M3 Max | `gpt-oss:120b` (65GB) | 자유 | 여유 |

   - 계산법: 모델 크기 + 3GB 정도가 메모리의 70% 안에 들어오면 됩니다. macOS가 통합 메모리의 약 70~75%만 GPU에 쓰게 하기 때문입니다.
   - 속도는 메모리 대역폭이 좌우합니다. M3 100GB/s, M3 Pro 150GB/s, M3 Max 300~400GB/s.
   - iMac과 MacBook은 같은 M3 칩이라 표가 같습니다. 팬이 없는 MacBook Air는 오래 돌리면 느려집니다.
   - Embedding 모델 `bge-m3`(1.2GB)는 모든 기종에서 됩니다.
3. **uv(유브이) 설치**: 파이썬 패키지를 자동으로 챙겨 주는 도구입니다.
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   코드 맨 위 주석에 필요한 패키지가 적혀 있어서 `uv run`만 하면 알아서 설치됩니다.

## 1. MCP Inspector로 먼저 테스트

AI 없이 도구가 잘 도는지 브라우저에서 직접 눌러 볼 수 있습니다.

```bash
cd lab
npx @modelcontextprotocol/inspector uv run memory_mcp.py
```

열린 화면에서 Connect → Tools → `remember`에 아무 문장이나 넣고 실행 → `recall`로 찾아보세요.

## 2. Claude에 연결

**Claude Code**

```bash
claude mcp add memory -- uv run /절대경로/mcp-lab/memory_mcp.py
claude mcp add images -- uv run /절대경로/mcp-lab/image_mcp.py
```

**Claude Desktop**: 설정 → 개발자 → 설정 편집에서 `claude_desktop_config.json`에 추가

```json
{
  "mcpServers": {
    "memory": { "command": "uv", "args": ["run", "/절대경로/mcp-lab/memory_mcp.py"] },
    "images": { "command": "uv", "args": ["run", "/절대경로/mcp-lab/image_mcp.py"] }
  }
}
```

`uv`를 못 찾는다는 오류가 나면 `"command"`에 `which uv`로 나온 전체 경로를 넣으세요.
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
      "memory": { "type": "local", "command": ["uv", "run", "/절대경로/mcp-lab/memory_mcp.py"], "codemode": false }
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
- "~/Downloads/xray 폴더 이미지를 '공개 데이터셋, 흉부'라는 메모로 등록해 줘"
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
