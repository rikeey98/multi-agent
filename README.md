# SOC Automation Multi-Agent System

리눅스 환경에서 SOC (System-on-Chip) 검증 시뮬레이션 중 발생하는 에러를 자동으로 감지, 분석, 처리하는 Multi-Agent 자동화 시스템입니다.

## Features

- **자동 에러 감지**: 로그 파일에서 에러를 자동으로 감지하고 분류
- **SOP 기반 해결**: SOP 문서를 검색하여 검증된 해결책 제시
- **자동 데이터 수집**: 로그, DB, 시스템 상태 정보 자동 수집
- **지능형 의사결정**: AI Agent가 모든 정보를 종합하여 최적의 해결 계획 수립
- **안전한 자동 실행**: 백업 및 롤백 지원으로 안전한 자동 수정
- **통합 알림**: DB INSERT를 통한 사내 메신저/메일 알림

## Architecture

### Multi-Agent System

시스템은 7개의 전문화된 Agent로 구성됩니다:

1. **Error Analyzer Agent**: 에러 분석 및 분류 (TIMEOUT/MEMORY/CONFIG 등)
2. **SOP Searcher Agent**: SOP 문서 검색 및 해결 절차 추출
3. **Data Collector Agent**: 로그, DB, 시스템 상태 수집
4. **Decision Maker Agent**: 정보 종합 및 실행 계획 수립
5. **Auto Executor Agent**: 안전한 자동 실행 (백업/롤백)
6. **Notification Agent**: 알림 메시지 생성 및 DB INSERT
7. **Supervisor Agent**: 전체 워크플로우 제어 (LangGraph)

### Technology Stack

- **LangChain**: Agent 프레임워크
- **LangGraph**: Multi-Agent 워크플로우 오케스트레이션
- **OpenAI GPT**: 언어 모델
- **MCP (Model Context Protocol)**: MongoDB, Oracle, Linux tools 통합
- **Python 3.10+**: 프로그래밍 언어

## Project Structure

```
soc_automation/
├── config/
│   ├── settings.py              # 전역 설정
│   ├── agent_prompts.py         # Agent별 프롬프트
│   └── mcp_config.json          # MCP 서버 설정
├── agents/
│   ├── error_analyzer.py        # 에러 분석 Agent
│   ├── sop_searcher.py          # SOP 검색 Agent
│   ├── data_collector.py        # 데이터 수집 Agent
│   ├── decision_maker.py        # 의사결정 Agent
│   ├── auto_executor.py         # 자동 실행 Agent
│   └── notification.py          # 알림 Agent
├── utils/
│   ├── state.py                 # LangGraph State 정의
│   ├── error_patterns.py        # 에러 패턴 정의
│   └── logger.py                # 로깅 설정
├── tools/
│   ├── file_tools.py            # 파일 시스템 Tools
│   ├── log_tools.py             # 로그 분석 Tools
│   ├── db_tools.py              # DB 관련 Tools
│   └── system_tools.py          # 시스템 명령 Tools
├── mcp_servers/
│   ├── mongodb_server.py        # MongoDB MCP
│   ├── oracle_server.py         # Oracle MCP
│   └── linux_tools_server.py    # Linux 명령 MCP
└── tests/
    ├── test_error_analyzer.py
    └── test_integration.py
```

## Installation

### Prerequisites

- Python 3.10 or higher
- uv (Python package manager)
- OpenAI API key
- MongoDB (optional)
- Oracle Database (optional)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd multi-agent
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Set up required directories:
```bash
sudo mkdir -p /var/log/sim /opt/sop /var/backup/soc /etc/soc
sudo chown $USER:$USER /var/log/sim /opt/sop /var/backup/soc /etc/soc
```

## Configuration

### Environment Variables

Edit `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-your-api-key-here

# Optional (with defaults)
OPENAI_MODEL=gpt-4o-mini
LOG_DIR=/var/log/sim
SOP_DIR=/opt/sop
```

See `.env.example` for all configuration options.

## Usage

### Basic Usage

Analyze a log file:

```bash
uv run python main.py /path/to/simulation.log
```

### With Custom Model

```bash
uv run python main.py /path/to/simulation.log --model gpt-4o
```

### Test Individual Agents

Each agent can be tested independently:

```bash
# Test Error Analyzer
uv run python -m soc_automation.agents.error_analyzer

# Test SOP Searcher
uv run python -m soc_automation.agents.sop_searcher

# Test Data Collector
uv run python -m soc_automation.agents.data_collector

# Test Decision Maker
uv run python -m soc_automation.agents.decision_maker

# Test Auto Executor
uv run python -m soc_automation.agents.auto_executor

# Test Notification
uv run python -m soc_automation.agents.notification
```

### Test Configuration

```bash
# Test settings loading
uv run python -m soc_automation.config.settings

# Test agent prompts
uv run python -m soc_automation.config.agent_prompts

# Test error patterns
uv run python -m soc_automation.utils.error_patterns

# Test state management
uv run python -m soc_automation.utils.state

# Test logging
uv run python -m soc_automation.utils.logger
```

## Workflow

The system follows this workflow:

```
1. Error Analyzer
   ↓
2. SOP Searcher (parallel)
   Data Collector (parallel)
   ↓
3. Decision Maker
   ↓
4. Auto Executor (if approved) → Notification
   OR
   Notification (if manual review required)
```

## Error Categories

The system classifies errors into the following categories:

- **TIMEOUT**: Simulation timeout errors
- **MEMORY**: Memory-related errors (OOM, leaks, segfaults)
- **CONFIG**: Configuration errors
- **ASSERTION**: Assertion failures
- **PROTOCOL**: Protocol violation errors
- **COMPILATION**: Compilation/elaboration errors
- **RUNTIME**: Runtime errors
- **UNKNOWN**: Unclassified errors

## Safety Features

### Auto-Execution Safety

- **Backup before modification**: All files are backed up before changes
- **Risk assessment**: Each operation is assessed for risk level
- **Conservative defaults**: Auto-execution only for low-risk operations
- **Rollback support**: All operations can be rolled back
- **Manual approval**: High-risk operations require manual approval

### Supported Auto-Execution Operations

✅ **Safe operations**:
- Configuration parameter updates (with backup)
- Log file cleanup
- Process restarts
- Cache clearing
- Environment variable updates

❌ **Manual-only operations**:
- Database schema changes
- System configuration changes
- File deletions (except temp files)
- Network changes

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest soc_automation/tests/test_error_analyzer.py

# Run with coverage
uv run pytest --cov=soc_automation
```

### Adding New Error Patterns

Edit `soc_automation/utils/error_patterns.py`:

```python
new_pattern = ErrorPattern(
    pattern_id="CUSTOM_001",
    category=ErrorCategory.CUSTOM,
    regex=r"your regex pattern here",
    severity=7,
    description="Pattern description",
    keywords=["keyword1", "keyword2"],
    suggested_action="Suggested resolution"
)
```

### Creating Custom Agents

Each agent module includes a template:

```python
from langgraph.prebuilt import create_react_agent

def create_custom_agent(llm, tools):
    return create_react_agent(
        model=llm,
        tools=tools,
        state_modifier="Your agent prompt here"
    )
```

## Logging

Logs are written to:
- Console: Colored, timestamped output
- `logs/soc_automation.log`: All log levels
- `logs/soc_automation_error.log`: Errors only
- `logs/soc_automation.workflow.<workflow_id>.log`: Per-workflow logs

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not set"**
   - Solution: Copy `.env.example` to `.env` and set your API key

2. **"Log file not found"**
   - Solution: Ensure the log file path is correct and accessible

3. **"MongoDB connection failed"**
   - Solution: Check MongoDB URI in `.env` or disable MongoDB features

4. **Agent test fails**
   - Solution: Ensure `.env` is configured with valid API key

## Future Enhancements

- [ ] MCP server implementations (MongoDB, Oracle, Linux tools)
- [ ] Web dashboard for monitoring
- [ ] Slack/Teams integration
- [ ] Historical error analysis
- [ ] Machine learning for pattern detection
- [ ] Distributed execution support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Your License Here]

## Contact

For questions or support, contact: [Your Contact Info]

---

**Note**: This is a Phase 1 implementation with 7 core agents. Future phases will add MCP server integrations, advanced analytics, and web interface.
