# MongoDB MCP Server

FastMCP 기반 MongoDB MCP 서버입니다.

## 기능

### Resources
- `mongodb://databases` - 데이터베이스 목록 조회
- `mongodb://{db}/collections` - 컬렉션 목록 조회
- `mongodb://{db}/{collection}/{id}` - 특정 문서 조회

### Tools
- `find` - 문서 검색 (filter, limit, projection)
- `insert_one` - 단일 문서 삽입
- `aggregate` - 읽기 전용 집계 파이프라인

## 설정

### 1. 환경 변수 설정

`.env` 파일에 MongoDB URI 추가:

```bash
MONGODB_URI=mongodb://localhost:27017
```

### 2. MCP 서버 활성화

`soc_automation/config/mcp_servers.json`에서 `enabled`를 `true`로 설정:

```json
{
  "name": "mongodb",
  "enabled": true
}
```

### 3. 서버 실행

독립 실행:
```bash
uv run python -m soc_automation.mcp_servers.mongodb_mcp
```

또는 main.py에서 자동으로 로드됩니다.

## 사용 예제

### Resource 조회

```python
# 데이터베이스 목록
mongodb://databases

# 컬렉션 목록
mongodb://soc_db/collections

# 특정 문서
mongodb://soc_db/errors/507f1f77bcf86cd799439011
```

### Tool 사용

#### find - 문서 검색
```python
{
  "database": "soc_db",
  "collection": "errors",
  "filter": {"severity": {"$gte": 8}},
  "limit": 10
}
```

#### insert_one - 문서 삽입
```python
{
  "database": "soc_db",
  "collection": "errors",
  "document": {
    "error_type": "TIMEOUT",
    "severity": 9,
    "timestamp": "2024-12-24T10:00:00"
  }
}
```

#### aggregate - 집계
```python
{
  "database": "soc_db",
  "collection": "errors",
  "pipeline": [
    {"$match": {"severity": {"$gte": 8}}},
    {"$group": {"_id": "$error_type", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
  ]
}
```

## 타입 변환

- **ObjectId**: 자동으로 string ↔ ObjectId 변환
- **datetime**: ISO 8601 형식으로 직렬화
- **BSON**: JSON으로 자동 변환

## 에러 처리

모든 에러는 JSON 형식으로 반환됩니다:

```json
{
  "error": "Error message here",
  "database": "soc_db",
  "collection": "errors"
}
```

## 보안

- **읽기 전용 집계**: `$out`, `$merge` 차단
- **수정/삭제 금지**: update, delete 도구 없음
- **쿼리 제한**: find 최대 100개 문서

## 개발자 노트

- Motor (비동기 MongoDB 드라이버) 사용
- FastMCP 데코레이터 기반 API
- 간단하고 명료한 코드 구조
