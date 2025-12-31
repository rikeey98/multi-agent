# SOC Automation Multi-Agent System - Documentation

이 디렉토리는 SOC 검증 자동화 Multi-Agent 시스템의 문서를 포함합니다.

## 📚 문서 목록

### 시스템 문서

#### [ARCHITECTURE.md](./ARCHITECTURE.md) - 시스템 아키텍처
전체 시스템 구조, 데이터 흐름, 컴포넌트 설명을 포함한 상세 아키텍처 문서

**포함 내용:**
- 🏗️ 전체 시스템 구조 다이어그램
- 🔄 데이터 흐름 시퀀스 다이어그램
- 📁 프로젝트 파일 구조
- 🔍 RAG 프로세스 상세 설명
- 🧩 주요 컴포넌트 설명
- ⚙️ 환경 변수 가이드
- 🚀 실행 방법
- 🔧 트러블슈팅

**대상 독자:** 시스템 아키텍처를 이해하려는 개발자, 유지보수 담당자

---

#### [BEGINNER_GUIDE.md](../BEGINNER_GUIDE.md) - 초보자 가이드
Python 초보자도 이해할 수 있도록 작성된 프로젝트 입문 가이드

**포함 내용:**
- 📖 프로젝트 소개
- 📂 디렉토리 구조 상세 설명
- 💾 설치 가이드
- 🎯 기본 개념 설명 (Agents, LangChain, Workflow, State, MCP)
- 📝 파일별 상세 설명
- ▶️ 실행 방법
- ⚙️ 설정 파일 가이드
- ❓ FAQ 및 문제 해결
- 📚 학습 자료

**대상 독자:** Python 초보자, 프로젝트를 처음 접하는 개발자

---

### 컴포넌트별 문서

#### [data/README.md](../soc_automation/data/README.md) - FAISS RAG 가이드
RAG (Retrieval-Augmented Generation) 설정 및 사용 가이드

**포함 내용:**
- 🗂️ FAISS 디렉토리 구조
- 🔧 FAISS Index 생성 방법
- 📝 에러 패턴 문서 포맷
- ⚙️ 환경 변수 설정
- 🌐 오프라인 서버 지원
- 📊 성능 최적화 팁
- 🔍 트러블슈팅

**대상 독자:** RAG 기능을 설정하거나 사용하는 개발자

---

## 🗺️ 문서 읽기 순서

### 처음 시작하는 경우
1. **[BEGINNER_GUIDE.md](../BEGINNER_GUIDE.md)** - 프로젝트 전체 이해
2. **[ARCHITECTURE.md](./ARCHITECTURE.md)** - 시스템 구조 파악
3. **[data/README.md](../soc_automation/data/README.md)** - RAG 설정 (선택사항)

### 시스템 개발/유지보수
1. **[ARCHITECTURE.md](./ARCHITECTURE.md)** - 아키텍처 참조
2. 각 컴포넌트별 소스 코드 주석
3. **[data/README.md](../soc_automation/data/README.md)** - RAG 관련 작업 시

### 문제 해결
1. **[ARCHITECTURE.md](./ARCHITECTURE.md)** - 트러블슈팅 섹션
2. **[BEGINNER_GUIDE.md](../BEGINNER_GUIDE.md)** - FAQ 섹션
3. **[data/README.md](../soc_automation/data/README.md)** - RAG 관련 문제

---

## 📋 주요 다이어그램

### 전체 시스템 구조
```
User → Main Workflow → 6 Agents (순차 실행)
                    ↓
           Error Analyzer (3 Sub-Agents)
                    ↓
              Workflow Storage
```

### Error Analyzer Sub-Agents
```
1. Pattern Matcher (+ RAG) → pattern_code, base_severity
2. Severity Assessor       → final_severity, modifiers
3. Root Cause Analyzer     → hypothesis, recommendations
```

### RAG 프로세스
```
Error Message → FAISS Search → Retrieved Docs → LLM → Pattern Match
```

전체 다이어그램은 [ARCHITECTURE.md](./ARCHITECTURE.md)를 참조하세요.

---

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# .env 파일 생성
cp .env.example .env

# 환경 변수 설정
vim .env
```

### 2. 의존성 설치
```bash
uv sync
```

### 3. 실행
```bash
# 전체 workflow 실행
uv run python main.py /path/to/log/file.log

# Error Analyzer만 테스트
uv run python -m soc_automation.agents.error_analyzer.analyzer
```

자세한 내용은 [BEGINNER_GUIDE.md](../BEGINNER_GUIDE.md)를 참조하세요.

---

## 🔧 주요 설정 파일

### `.env` - 환경 변수
```bash
OPENAI_API_KEY=sk-your-api-key
OPENAI_BASE_URL=https://your-api.com/v1
OPENAI_EMBEDDING_BASE_URL=https://your-embedding-api.com/v1
MONGODB_URI=mongodb://localhost:27017
```

### `mcp_servers.json` - MCP 서버 설정
```json
{
  "mcpServers": {
    "mongodb": {
      "command": "uv",
      "args": ["run", "python", "-m", "soc_automation.mcp_servers.mongodb_mcp"]
    }
  }
}
```

### `config/prompts/*.md` - Agent 프롬프트
각 Agent의 system prompt를 Markdown 파일로 관리

---

## 🎯 주요 개념

### Agent
특정 작업을 수행하는 독립적인 모듈
- Error Analyzer: 에러 분석
- SOP Searcher: SOP 검색
- Decision Maker: 의사결정
- 등...

### Sub-Agent
복잡한 Agent를 세분화한 전문화된 모듈
- Pattern Matcher: 패턴 매칭
- Severity Assessor: 심각도 평가
- Root Cause Analyzer: 근본 원인 분석

### RAG (Retrieval-Augmented Generation)
과거 데이터를 검색하여 LLM 응답 품질 향상
- FAISS vector store
- Semantic search
- MMR (Maximal Marginal Relevance)

### Workflow Storage
실행 결과를 workflow_id별로 저장
- 완전한 추적성
- 감사 가능
- 디버깅 용이

---

## 🔍 디버깅 팁

### 로그 확인
```bash
# 최근 workflow 로그
ls -lt workflows/

# 특정 workflow 결과
cat workflows/{workflow_id}/complete.json | jq .

# RAG 사용 여부 확인
cat workflows/{workflow_id}/error_analyzer.json | jq '.pattern_result.rag_used'
```

### Agent 단독 실행
```bash
# Error Analyzer
uv run python -m soc_automation.agents.error_analyzer.analyzer

# Pattern Matcher
uv run python -m soc_automation.agents.error_analyzer.sub_agents.pattern_matcher
```

### 환경 변수 확인
```bash
# .env 파일 확인
cat .env

# 특정 변수 확인
grep OPENAI_API_KEY .env
```

---

## 📊 성능 모니터링

### Workflow 실행 시간
- Error Analyzer: ~10-30초 (RAG 포함)
- 전체 Workflow: ~1-3분

### RAG 성능
- FAISS 검색: ~100ms
- Embedding 생성: ~200ms
- LLM 응답: ~2-5초

### 최적화 팁
1. FAISS index 크기: < 100K 문서
2. Chunk size: 1000 (패턴 길이에 따라 조정)
3. MMR parameters: k=10, fetch_k=30

---

## 🤝 기여 가이드

### 코드 스타일
- Python 3.10+
- Type hints 사용
- Docstrings 작성 (Google style)
- 상대 임포트 대신 절대 임포트 사용

### 테스트
- 각 Agent는 독립 실행 가능해야 함
- `__main__` 블록으로 테스트 코드 작성
- 샘플 데이터로 테스트

### 문서
- 새 기능 추가 시 관련 문서 업데이트
- 다이어그램이 필요한 경우 Mermaid 사용
- 예제 코드 포함

---

## 📞 지원

### 문제 발생 시
1. [ARCHITECTURE.md](./ARCHITECTURE.md) - 트러블슈팅 섹션 확인
2. [BEGINNER_GUIDE.md](../BEGINNER_GUIDE.md) - FAQ 섹션 확인
3. 로그 파일 확인
4. GitHub Issues 등록

### 학습 자료
- [LangChain 공식 문서](https://python.langchain.com/)
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [FastMCP 문서](https://github.com/jlowin/fastmcp)
- [FAISS 문서](https://github.com/facebookresearch/faiss)

---

## 📝 변경 이력

### v1.0.0 (Current)
- ✅ Error Analyzer Sub-Agent 구조 구현
- ✅ RAG (FAISS) 통합
- ✅ Workflow Storage 구현
- ✅ MCP MongoDB 서버 통합
- ✅ 문서화 완료

---

## 📄 라이선스

이 프로젝트의 라이선스 정보는 프로젝트 루트의 LICENSE 파일을 참조하세요.

---

**마지막 업데이트**: 2024-12-28
