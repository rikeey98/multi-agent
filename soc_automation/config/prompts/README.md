# Agent Prompts

이 디렉토리에는 각 Agent의 시스템 프롬프트가 개별 마크다운 파일로 저장되어 있습니다.

## 프롬프트 파일 목록

- **supervisor.md** - Supervisor Agent (워크플로우 조정)
- **error_analyzer.md** - Error Analyzer Agent (에러 분석 및 분류)
- **sop_searcher.md** - SOP Searcher Agent (SOP 문서 검색)
- **data_collector.md** - Data Collector Agent (데이터 수집)
- **decision_maker.md** - Decision Maker Agent (의사결정)
- **auto_executor.md** - Auto Executor Agent (자동 실행)
- **notification.md** - Notification Agent (알림 생성)

## 프롬프트 수정 방법

1. 해당 Agent의 MD 파일을 직접 수정
2. 마크다운 형식으로 작성
3. 변경 사항이 즉시 반영됨 (재시작 필요 없음)

## 프롬프트 구조

각 프롬프트는 다음 구조를 따릅니다:

```markdown
# Agent Name

You are the [Agent Name] for SOC verification automation.

## Your Mission
[Agent의 주요 목적]

## [섹션 제목]
[상세 내용]

## Output Format
[기대되는 출력 형식]
```

## 사용 예제

```python
from soc_automation.config.agent_prompts import get_agent_prompt

# 프롬프트 로드
prompt = get_agent_prompt("error_analyzer")

# Agent 생성 시 사용
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=prompt
)
```

## 프롬프트 테스트

```bash
# 모든 프롬프트 확인
uv run python soc_automation/config/agent_prompts.py

# 특정 프롬프트 확인
uv run python -c "from soc_automation.config.agent_prompts import get_agent_prompt; print(get_agent_prompt('error_analyzer'))"
```

## 주의사항

- 프롬프트 파일명은 `agent_prompts.py`의 `AVAILABLE_AGENTS` 리스트와 일치해야 함
- 마크다운 형식을 유지하면서 자유롭게 수정 가능
- Git으로 버전 관리됨
- 프롬프트 변경은 즉시 반영되므로 신중하게 수정
