# SOC 자동화 Multi-Agent 시스템 - 초보자 가이드

## 📖 목차

1. [프로젝트 소개](#프로젝트-소개)
2. [전체 구조](#전체-구조)
3. [설치 방법](#설치-방법)
4. [기본 개념](#기본-개념)
5. [프로젝트 구조 상세](#프로젝트-구조-상세)
6. [실행 방법](#실행-방법)
7. [결과 확인](#결과-확인)
8. [설정 파일](#설정-파일)
9. [문제 해결](#문제-해결)

---

## 프로젝트 소개

### 이 프로젝트는 무엇인가요?

SOC(System-on-Chip) 검증 작업 중 발생하는 에러를 **자동으로 분석하고 해결**하는 시스템입니다.

### 어떻게 동작하나요?

마치 여러 전문가들이 협력하는 것처럼, **7개의 AI Agent**가 각자의 역할을 수행합니다:

```
1. 에러 분석가 (Error Analyzer)
   └─> 로그 파일에서 에러를 찾아 분석합니다

2. SOP 검색가 (SOP Searcher)
   └─> 해결 방법 문서를 검색합니다

3. 데이터 수집가 (Data Collector)
   └─> 시스템 상태와 로그를 수집합니다

4. 의사결정자 (Decision Maker)
   └─> 모든 정보를 종합해서 해결 방법을 결정합니다

5. 자동 실행자 (Auto Executor)
   └─> 승인된 작업을 자동으로 실행합니다

6. 알림 담당 (Notification)
   └─> 결과를 알려줍니다
```

### 왜 필요한가요?

- ⏱️ **시간 절약**: 사람이 수동으로 분석하던 작업을 자동화
- 🎯 **정확성**: 패턴 기반 에러 분석으로 실수 방지
- 📊 **추적 가능**: 모든 과정이 기록되어 나중에 확인 가능
- 🔄 **재사용**: 한번 해결한 문제는 다음에도 자동 해결

---

## 전체 구조

### 프로젝트 디렉토리 구조

```
multi-agent/
├── main.py                        # 프로그램 시작 파일
├── pyproject.toml                 # 필요한 라이브러리 목록
├── .env                           # 설정 (API 키 등)
├── workflows/                     # 실행 결과 저장 폴더
│
├── soc_automation/                # 메인 코드
│   ├── agents/                    # 6개 Agent 코드
│   │   ├── error_analyzer.py     # 에러 분석
│   │   ├── sop_searcher.py       # SOP 검색
│   │   ├── data_collector.py     # 데이터 수집
│   │   ├── decision_maker.py     # 의사결정
│   │   ├── auto_executor.py      # 자동 실행
│   │   └── notification.py       # 알림
│   │
│   ├── config/                    # 설정 파일들
│   │   ├── settings.py           # 전역 설정
│   │   ├── agent_prompts.py      # Agent 프롬프트 로더
│   │   ├── mcp_servers.json      # MCP 서버 설정
│   │   └── prompts/              # Agent별 프롬프트 (MD 파일)
│   │       ├── error_analyzer.md
│   │       ├── sop_searcher.md
│   │       └── ...
│   │
│   ├── utils/                     # 도구 모음
│   │   ├── state.py              # 상태 관리
│   │   ├── logger.py             # 로그 기록
│   │   ├── error_patterns.py     # 에러 패턴
│   │   └── workflow_storage.py   # 결과 저장
│   │
│   └── mcp_servers/               # MCP 서버
│       └── mongodb_mcp.py        # MongoDB MCP 서버
│
└── README.md                      # 프로젝트 설명
```

---

## 설치 방법

### 1단계: 필수 프로그램 설치

#### Python 3.10 이상 설치
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10

# 확인
python3 --version
```

#### uv 패키지 매니저 설치
```bash
# uv는 Python 라이브러리를 빠르게 설치해주는 도구입니다
curl -LsSf https://astral.sh/uv/install.sh | sh

# 확인
uv --version
```

### 2단계: 프로젝트 복제

```bash
# 프로젝트 다운로드
git clone <repository-url>
cd multi-agent

# 라이브러리 설치
uv sync
```

### 3단계: 환경 설정

```bash
# .env.example을 복사해서 .env 파일 생성
cp .env.example .env

# .env 파일 수정
vim .env  # 또는 원하는 에디터 사용
```

**.env 파일에서 꼭 설정해야 할 것:**

```bash
# OpenAI API 키 (필수)
OPENAI_API_KEY=sk-your-api-key-here

# 선택사항: 커스텀 엔드포인트 사용 시
OPENAI_BASE_URL=https://api.openai.com/v1

# MongoDB 연결 (선택)
MONGODB_URI=mongodb://localhost:27017
```

---

## 기본 개념

### 1. Agent란?

**Agent**는 특정 작업을 수행하는 AI 프로그램입니다.

```python
# 예시: Error Analyzer Agent
agent = create_error_analyzer_agent(
    llm=ChatOpenAI(),           # AI 모델
    tools=[read_log, analyze]   # 사용할 도구들
)

# Agent에게 작업 요청
result = agent.invoke("로그 파일을 분석해줘")
```

### 2. LangChain과 LangGraph

- **LangChain**: AI Agent를 만드는 도구
- **LangGraph**: 여러 Agent를 연결하는 도구

```
[에러 분석] → [SOP 검색] → [의사결정] → [실행]
    ↓            ↓
    └─────→ [데이터 수집]
```

### 3. Workflow (워크플로우)

여러 Agent가 순서대로 실행되는 과정입니다.

```python
# Workflow 예시
workflow = StateGraph()
workflow.add_node("error_analyzer", error_analyzer_node)
workflow.add_node("sop_searcher", sop_searcher_node)
workflow.add_edge("error_analyzer", "sop_searcher")
```

### 4. State (상태)

각 Agent가 공유하는 정보입니다.

```python
state = {
    "workflow_id": "workflow-123",
    "log_file_path": "/var/log/error.log",
    "error_analysis": {...},      # 에러 분석 결과
    "sop_results": [...],         # SOP 검색 결과
    "resolution_plan": {...}      # 해결 계획
}
```

### 5. MCP (Model Context Protocol)

외부 데이터베이스나 도구를 Agent에 연결하는 방법입니다.

```
Agent → MCP Server → MongoDB/Oracle
```

---

## 프로젝트 구조 상세

### 📁 agents/ - Agent 코드

각 Agent는 **3가지 요소**로 구성됩니다:

#### 1. Tools (도구)
```python
@tool
def read_log_file(file_path: str) -> str:
    """로그 파일을 읽는 도구"""
    with open(file_path, 'r') as f:
        return f.read()
```

#### 2. Agent 생성 함수
```python
def create_error_analyzer_agent(llm, tools):
    """에러 분석 Agent 생성"""
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=get_agent_prompt("error_analyzer")
    )
```

#### 3. 실행 함수
```python
async def run_error_analyzer(llm, log_file_path):
    """에러 분석 Agent 실행"""
    agent = create_error_analyzer_agent(llm, DEFAULT_TOOLS)
    result = await agent.ainvoke({"messages": [...]})
    return result
```

### 📁 config/ - 설정 파일

#### settings.py
전역 설정을 관리합니다.

```python
from soc_automation.config.settings import settings

# 사용 예
model = settings.openai.model           # "gpt-4o-mini"
api_key = settings.openai.api_key       # API 키
log_dir = settings.paths.log_dir        # "/var/log/sim"
```

#### prompts/ - Agent 프롬프트

각 Agent의 역할과 지시사항이 마크다운 파일로 저장되어 있습니다.

**예시: error_analyzer.md**
```markdown
# Error Analyzer Agent

You are the Error Analyzer Agent...

## Your Mission
Analyze simulation errors from log files...

## Error Categories
- TIMEOUT: Simulation timeout errors
- MEMORY: Memory-related errors
...
```

### 📁 utils/ - 유틸리티

#### state.py - 상태 관리
```python
# 초기 상태 생성
initial_state = create_initial_state(
    log_file_path="/var/log/error.log",
    trigger_event="cli"
)

# 상태 업데이트
state = update_state_with_error_analysis(state, error_analysis)
```

#### logger.py - 로깅
```python
from soc_automation.utils.logger import get_logger

logger = get_logger()
logger.info("작업 시작")
logger.error("에러 발생!")
```

#### workflow_storage.py - 결과 저장
```python
from soc_automation.utils.workflow_storage import get_workflow_storage

storage = get_workflow_storage()
storage.save_step(
    workflow_id="workflow-123",
    step_name="error_analyzer",
    data={"result": "..."}
)
```

### 📁 mcp_servers/ - MCP 서버

MongoDB와 통신하는 MCP 서버가 포함되어 있습니다.

```python
# MongoDB 데이터베이스 목록 조회
mongodb://databases

# 특정 문서 조회
mongodb://soc_db/errors/123

# 문서 검색
find(database="soc_db", collection="errors", filter={"severity": 8})
```

---

## 실행 방법

### 기본 실행

```bash
# 로그 파일 분석
uv run python main.py --log-file /path/to/error.log
```

### 옵션

```bash
# MCP 서버 없이 실행
uv run python main.py --log-file error.log --no-mcp

# 도움말 보기
uv run python main.py --help
```

### 실행 과정

```
1. main.py 시작
   ↓
2. 설정 로드 (.env, settings.py)
   ↓
3. MCP 서버 연결 (선택사항)
   ↓
4. Workflow 생성
   ↓
5. Agent들 순차 실행
   - Error Analyzer
   - SOP Searcher
   - Data Collector
   - Decision Maker
   - Auto Executor (조건부)
   - Notification
   ↓
6. 결과 저장 (workflows/ 디렉토리)
   ↓
7. 완료
```

---

## 결과 확인

### 저장 위치

모든 실행 결과는 `workflows/` 디렉토리에 저장됩니다.

```
workflows/
└── workflow-20241224-103045-abc123/
    ├── summary.txt              # 요약 (사람이 읽기 쉬운 형식)
    ├── complete.json            # 전체 결과
    ├── error_analyzer.json      # 에러 분석 결과
    ├── sop_searcher.json        # SOP 검색 결과
    ├── data_collector.json      # 수집 데이터
    ├── decision_maker.json      # 의사결정 내용
    ├── auto_executor.json       # 실행 결과
    └── notification.json        # 알림 정보
```

### summary.txt 예시

```
=== Workflow Summary: workflow-20241224-103045-abc123 ===
Completed: 2024-12-24 10:35:23

--- Trigger ---
Event: cli
Log File: /var/log/sim/error.log

--- Error Analysis ---
Error Type: TIMEOUT
Severity: 8/10
Message: Simulation timeout after 3600 seconds
Location: testbench.sv:145

--- Resolution Plan ---
Auto-executable: false
Risk Level: MEDIUM
Root Cause: Timeout configuration too low

Resolution Steps:
  1. Increase timeout to 7200
  2. Rerun simulation
```

### JSON 파일 확인

```bash
# 예쁘게 출력
cat workflows/workflow-123/error_analyzer.json | python -m json.tool

# 특정 필드만 확인 (jq 사용)
cat workflows/workflow-123/complete.json | jq '.error_analysis.severity'
```

---

## 설정 파일

### .env - 환경 변수

```bash
# OpenAI 설정
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=                    # 선택사항
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7

# MongoDB 설정 (선택)
MONGODB_URI=mongodb://localhost:27017

# 경로 설정
LOG_DIR=/var/log/sim
SOP_DIR=/opt/sop
BACKUP_DIR=/var/backup/soc

# 알림 설정
NOTIFICATION_ENABLED=true
NOTIFICATION_CRITICAL_THRESHOLD=8

# 디버그 모드
DEBUG=false
```

### mcp_servers.json - MCP 서버 설정

```json
{
  "servers": [
    {
      "name": "mongodb",
      "command": "uv",
      "args": ["run", "python", "-m", "soc_automation.mcp_servers.mongodb_mcp"],
      "env": {
        "MONGODB_URI": "${MONGODB_URI}"
      },
      "enabled": false
    }
  ]
}
```

**사용하려면:**
1. `"enabled": false`를 `"enabled": true`로 변경
2. MongoDB가 실행 중인지 확인

---

## 문제 해결

### Q: "OPENAI_API_KEY not set" 에러

**원인**: API 키가 설정되지 않음

**해결**:
```bash
# .env 파일 확인
cat .env | grep OPENAI_API_KEY

# 없으면 추가
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
```

### Q: "Module not found" 에러

**원인**: 라이브러리가 설치되지 않음

**해결**:
```bash
# 라이브러리 재설치
uv sync

# 특정 라이브러리 설치
uv add langchain
```

### Q: 실행이 너무 느려요

**원인**: MCP 서버 연결 시도

**해결**:
```bash
# MCP 없이 실행
uv run python main.py --log-file error.log --no-mcp
```

### Q: 결과가 어디에 저장되나요?

**답변**: `workflows/` 디렉토리에 저장됩니다.

```bash
# 최근 결과 확인
ls -lt workflows/ | head -5

# 특정 workflow 결과 보기
cat workflows/workflow-123/summary.txt
```

### Q: Agent 프롬프트를 수정하고 싶어요

**답변**: `soc_automation/config/prompts/` 디렉토리의 MD 파일을 수정하세요.

```bash
# 에러 분석 Agent 프롬프트 수정
vim soc_automation/config/prompts/error_analyzer.md

# 재시작 없이 바로 반영됩니다!
```

### Q: MongoDB가 필요한가요?

**답변**: 아니요, 선택사항입니다.

- MongoDB 없이도 기본 기능은 모두 동작합니다
- MongoDB는 과거 에러 이력 조회용으로만 사용됩니다

---

## 추가 학습 자료

### Python 기초
- [Python 공식 튜토리얼](https://docs.python.org/ko/3/tutorial/)
- [점프 투 파이썬](https://wikidocs.net/book/1)

### LangChain
- [LangChain 공식 문서](https://python.langchain.com/)
- [LangGraph 튜토리얼](https://langchain-ai.github.io/langgraph/)

### 비동기 프로그래밍 (async/await)
- [asyncio 가이드](https://docs.python.org/ko/3/library/asyncio.html)

---

## 프로젝트 기여

버그를 발견하거나 개선 아이디어가 있다면:

1. Issue 등록
2. Pull Request 제출
3. 문서 개선 제안

---

## 라이선스

이 프로젝트의 라이선스 정보는 LICENSE 파일을 참조하세요.

---

## 연락처

문의사항이나 도움이 필요하면 Issue를 등록해주세요.

**Happy Coding! 🎉**
