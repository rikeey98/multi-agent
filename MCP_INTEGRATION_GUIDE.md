# MCP 서버 통합 가이드

이 가이드는 사용자가 직접 만든 fastmcp 기반 MCP 서버를 SOC Automation 시스템에 통합하는 방법을 설명합니다.

## 목차

1. [개요](#개요)
2. [MCP 서버 설정하기](#mcp-서버-설정하기)
3. [MCP 서버 테스트하기](#mcp-서버-테스트하기)
4. [통합 워크플로우 실행](#통합-워크플로우-실행)
5. [트러블슈팅](#트러블슈팅)

## 개요

SOC Automation 시스템은 stdio 방식으로 MCP 서버와 통신합니다. 사용자가 만든 MongoDB, Oracle MCP 서버를 쉽게 통합할 수 있도록 설계되었습니다.

### 시스템 구조

```
SOC Automation System
    ↓
MCPClientLoader (soc_automation/mcp_client/)
    ↓
stdio connection
    ↓
Your MCP Servers (MongoDB, Oracle)
```

## MCP 서버 설정하기

### 1. 설정 파일 위치

MCP 서버 설정은 다음 파일에서 관리됩니다:

```
soc_automation/config/mcp_servers.json
```

### 2. 설정 파일 형식

```json
{
  "servers": [
    {
      "name": "mongodb",
      "description": "My MongoDB MCP Server",
      "command": "python",
      "args": ["-m", "your_mongodb_mcp_server"],
      "env": {
        "MONGODB_URI": "${MONGODB_URI}",
        "MONGODB_DB": "${MONGODB_DB}"
      },
      "enabled": true
    },
    {
      "name": "oracle",
      "description": "My Oracle MCP Server",
      "command": "python",
      "args": ["-m", "your_oracle_mcp_server"],
      "env": {
        "ORACLE_USER": "${ORACLE_USER}",
        "ORACLE_PASSWORD": "${ORACLE_PASSWORD}",
        "ORACLE_DSN": "${ORACLE_DSN}"
      },
      "enabled": true
    }
  ]
}
```

### 3. 필드 설명

| 필드 | 필수 | 설명 |
|------|------|------|
| `name` | ✅ | MCP 서버 이름 (로그에 표시됨) |
| `description` | ❌ | 서버 설명 |
| `command` | ✅ | 실행할 명령어 (예: `python`, `uvx`, `node`) |
| `args` | ❌ | 명령어 인자 배열 |
| `env` | ❌ | 환경 변수 (`.env`에서 자동 확장) |
| `enabled` | ❌ | 활성화 여부 (기본값: `true`) |

### 4. 환경 변수 사용

`${VAR_NAME}` 형식으로 `.env` 파일의 환경 변수를 참조할 수 있습니다:

```json
{
  "env": {
    "MONGODB_URI": "${MONGODB_URI}"
  }
}
```

`.env` 파일:
```bash
MONGODB_URI=mongodb://localhost:27017
```

## MCP 서버 추가 예제

### 예제 1: Python 모듈로 실행

당신의 MCP 서버가 Python 패키지인 경우:

```json
{
  "name": "my_mongodb",
  "command": "python",
  "args": ["-m", "my_mcp_servers.mongodb"],
  "env": {
    "MONGODB_URI": "${MONGODB_URI}"
  },
  "enabled": true
}
```

### 예제 2: 스크립트 파일로 실행

MCP 서버가 단일 스크립트 파일인 경우:

```json
{
  "name": "my_oracle",
  "command": "python",
  "args": ["/home/user/mcp_servers/oracle_server.py"],
  "env": {
    "ORACLE_USER": "${ORACLE_USER}",
    "ORACLE_PASSWORD": "${ORACLE_PASSWORD}",
    "ORACLE_DSN": "${ORACLE_DSN}"
  },
  "enabled": true
}
```

### 예제 3: uvx로 실행

uvx 패키지로 배포된 MCP 서버:

```json
{
  "name": "my_custom_mcp",
  "command": "uvx",
  "args": ["my-mcp-package", "--config", "/path/to/config.json"],
  "enabled": true
}
```

### 예제 4: 가상환경의 Python 사용

특정 가상환경의 Python을 사용하는 경우:

```json
{
  "name": "my_mongodb",
  "command": "/home/user/venvs/mcp/bin/python",
  "args": ["-m", "my_mcp_servers.mongodb"],
  "env": {
    "MONGODB_URI": "${MONGODB_URI}"
  },
  "enabled": true
}
```

## MCP 서버 테스트하기

### 1. MCP 클라이언트 단독 테스트

먼저 MCP 서버 연결을 테스트합니다:

```bash
# MCP 서버 로드 테스트
uv run python -m soc_automation.mcp_client
```

**예상 출력:**
```
=== MCP Client Loader Test ===

Loading MCP server: mongodb
  ✓ Loaded 5 tools from mongodb
    - query_collection
    - insert_document
    - update_document
    - delete_document
    - aggregate_pipeline

Loading MCP server: oracle
  ✓ Loaded 8 tools from oracle
    - execute_query
    - execute_procedure
    - ...

Loaded 13 tools from MCP servers

Available tools:
1. query_collection
   Description: Query MongoDB collection
2. insert_document
   Description: Insert document to MongoDB
...
```

### 2. 개별 MCP 서버 테스트

MCP 서버를 직접 실행해서 stdio 통신이 가능한지 확인:

```bash
# MongoDB MCP 서버 직접 실행
python -m your_mongodb_mcp_server

# Oracle MCP 서버 직접 실행
python -m your_oracle_mcp_server
```

서버가 정상적으로 시작되면 JSON-RPC 메시지를 주고받을 수 있어야 합니다.

### 3. 헬퍼 함수 테스트

```bash
# MCP 헬퍼 함수 테스트
uv run python -m soc_automation.mcp_client.helpers
```

## 통합 워크플로우 실행

### 1. MCP 도구와 함께 실행

```bash
# 기본 실행 (MCP 도구 활성화)
uv run python main.py /path/to/log/file.log
```

### 2. MCP 도구 없이 실행

```bash
# MCP 도구 비활성화
uv run python main.py /path/to/log/file.log --no-mcp
```

### 3. 로그 확인

실행 중 다음과 같은 로그를 확인할 수 있습니다:

```
Loading MCP tools...
Loading MCP server: mongodb
  ✓ Loaded 5 tools from mongodb
Loading MCP server: oracle
  ✓ Loaded 8 tools from oracle
Loaded 13 MCP tools
MCP Tools: Enabled
```

## 트러블슈팅

### 문제 1: "No MCP servers configured"

**원인:** `mcp_servers.json` 파일이 없거나 비어있음

**해결:**
```bash
# 설정 파일 확인
cat soc_automation/config/mcp_servers.json

# enabled가 true로 설정되어 있는지 확인
```

### 문제 2: "Failed to load MCP server"

**원인:** MCP 서버 실행 명령이 잘못되었거나 서버에 문제가 있음

**해결:**
1. MCP 서버를 직접 실행해보기:
   ```bash
   python -m your_mongodb_mcp_server
   ```

2. 명령어 경로 확인:
   ```bash
   which python
   ```

3. 환경 변수 확인:
   ```bash
   cat .env
   ```

### 문제 3: "ImportError: No module named ..."

**원인:** MCP 서버의 의존성이 설치되지 않음

**해결:**
```bash
# MCP 서버가 있는 환경에 의존성 설치
cd /path/to/your/mcp/server
pip install -r requirements.txt

# 또는 uv 사용
uv pip install -r requirements.txt
```

### 문제 4: 환경 변수가 확장되지 않음

**원인:** `.env` 파일에 변수가 정의되지 않음

**해결:**
```bash
# .env 파일 확인
cat .env

# 변수 추가
echo "MONGODB_URI=mongodb://localhost:27017" >> .env
```

### 문제 5: MCP 도구가 Agent에 전달되지 않음

**원인:** MCP 도구 로딩 실패 또는 비활성화됨

**해결:**
1. 로그 확인:
   ```bash
   cat logs/soc_automation.log | grep "MCP"
   ```

2. MCP 활성화 확인:
   ```bash
   # --no-mcp 플래그를 사용하지 않았는지 확인
   uv run python main.py /path/to/log.log
   ```

## 디버깅 팁

### 1. 상세 로그 활성화

`.env` 파일에서:
```bash
DEBUG=true
```

### 2. MCP 서버 로그 확인

MCP 서버에 로깅 추가:

```python
# your_mcp_server.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 3. 단계별 테스트

```bash
# 1단계: MCP 서버 단독 실행
python -m your_mongodb_mcp_server

# 2단계: MCP 클라이언트 테스트
uv run python -m soc_automation.mcp_client

# 3단계: 전체 워크플로우
uv run python main.py /path/to/log.log
```

## 고급 설정

### 여러 MCP 서버 동시 사용

```json
{
  "servers": [
    {
      "name": "mongodb_prod",
      "command": "python",
      "args": ["-m", "my_mcp.mongodb"],
      "env": {
        "MONGODB_URI": "mongodb://prod:27017"
      },
      "enabled": true
    },
    {
      "name": "mongodb_dev",
      "command": "python",
      "args": ["-m", "my_mcp.mongodb"],
      "env": {
        "MONGODB_URI": "mongodb://dev:27017"
      },
      "enabled": false
    },
    {
      "name": "oracle",
      "command": "python",
      "args": ["-m", "my_mcp.oracle"],
      "enabled": true
    }
  ]
}
```

### 조건부 MCP 서버 활성화

개발/운영 환경에 따라 다른 서버 사용:

```bash
# .env.development
MONGODB_URI=mongodb://localhost:27017

# .env.production
MONGODB_URI=mongodb://prod-server:27017
```

## 참고 자료

- [MCP 공식 문서](https://modelcontextprotocol.io/)
- [fastmcp 문서](https://github.com/jlowin/fastmcp)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)

## 예제 MCP 서버 템플릿

### MongoDB MCP 서버 예제

```python
# mongodb_mcp_server.py
from fastmcp import FastMCP
import pymongo

mcp = FastMCP("MongoDB Tools")

@mcp.tool()
def query_collection(collection: str, filter: dict) -> list:
    """Query MongoDB collection"""
    client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    results = list(db[collection].find(filter))
    return results

if __name__ == "__main__":
    mcp.run()
```

### Oracle MCP 서버 예제

```python
# oracle_mcp_server.py
from fastmcp import FastMCP
import oracledb

mcp = FastMCP("Oracle Tools")

@mcp.tool()
def execute_query(sql: str) -> list:
    """Execute Oracle SQL query"""
    connection = oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("ORACLE_DSN")
    )
    cursor = connection.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    return results

if __name__ == "__main__":
    mcp.run()
```

---

이제 당신의 MCP 서버를 `mcp_servers.json`에 추가하고 테스트해보세요!
