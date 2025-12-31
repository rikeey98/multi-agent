# Data Directory

This directory contains data files used by the SOC Automation system.

## FAISS Index for RAG (Retrieval-Augmented Generation)

The `faiss_index/` directory contains the FAISS vector store used by the Pattern Matcher sub-agent for RAG-based error analysis.

**Knowledge Base Contents:**
1. **Error Patterns**: 100+ company-specific error patterns with codes, keywords, and severity
2. **SOP Documents**: Standard Operating Procedures for error resolution with step-by-step instructions
3. **Past Solutions**: Historical successful resolutions and workarounds

This unified knowledge base enables the Error Analyzer to find patterns, retrieve SOPs, and learn from past solutions all in one RAG search.

### Directory Structure

```
soc_automation/data/
├── README.md
└── faiss_index/          # FAISS vector store
    ├── index.faiss       # FAISS index file
    └── index.pkl         # Metadata pickle file
```

### Creating the FAISS Index

To create a FAISS index from your error pattern documents:

```python
import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Set environment variables
os.environ["TIKTOKEN_CACHE_DIR"] = os.path.expanduser("~/.cache/tiktoken")
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Initialize embeddings
embeddings_kwargs = {
    "model": "text-embedding-ada-002",
    "openai_api_key": "your-api-key"
}

# Use dedicated embedding URL if different from LLM URL
embedding_base_url = os.getenv("OPENAI_EMBEDDING_BASE_URL", "your-embedding-api-url")  # Optional
if embedding_base_url:
    embeddings_kwargs["openai_api_base"] = embedding_base_url

embeddings = OpenAIEmbeddings(**embeddings_kwargs)

# Load documents: error patterns + SOPs + past solutions
# Combine all knowledge sources into one directory or load separately
loader = DirectoryLoader(
    "path/to/your/knowledge_base/",  # Contains: patterns/, sops/, solutions/
    glob="**/*.md",
    loader_cls=TextLoader
)
documents = loader.load()

# Or load multiple sources separately and combine
# pattern_loader = DirectoryLoader("path/to/patterns/", glob="**/*.md", loader_cls=TextLoader)
# sop_loader = DirectoryLoader("path/to/sops/", glob="**/*.md", loader_cls=TextLoader)
# solution_loader = DirectoryLoader("path/to/solutions/", glob="**/*.md", loader_cls=TextLoader)
# documents = pattern_loader.load() + sop_loader.load() + solution_loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# Create FAISS index
vector_store = FAISS.from_documents(splits, embeddings)

# Save to disk
faiss_index_path = Path(__file__).parent / "faiss_index"
vector_store.save_local(str(faiss_index_path))

print(f"FAISS index saved to: {faiss_index_path}")
```

### Using the FAISS Index

The Pattern Matcher sub-agent automatically loads the FAISS index if it exists:

```python
from soc_automation.agents.error_analyzer.sub_agents.pattern_matcher import run_pattern_matcher
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

result = await run_pattern_matcher(
    llm=llm,
    error_message="Cache coherency violation detected",
    log_context="Additional log context..."
)

# The pattern matcher will use RAG to search similar patterns
# from the FAISS index before making a match decision
```

### Configuration

Set the following environment variables in `.env`:

```bash
# Required
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Optional
OPENAI_BASE_URL=https://your-llm-api-endpoint.com/v1  # Base URL for LLM API
OPENAI_EMBEDDING_BASE_URL=https://your-embedding-api-endpoint.com/v1  # Base URL for embeddings API (if different)
```

**Note**: If `OPENAI_EMBEDDING_BASE_URL` is not set, the system will fall back to using `OPENAI_BASE_URL` for embeddings. This allows you to use separate API endpoints for LLM and embeddings if needed.

### Offline Server Support

For offline servers, the system automatically sets the TIKTOKEN cache directory:

```python
os.environ["TIKTOKEN_CACHE_DIR"] = os.path.expanduser("~/.cache/tiktoken")
```

This allows the tokenizer to work without internet access after the initial download.

### Updating the Index

To update the FAISS index with new error patterns:

1. Add new pattern documents to your source directory
2. Re-run the index creation script
3. The Pattern Matcher will automatically use the updated index on next run

### Fallback Behavior

If the FAISS index is not found or fails to load:
- The Pattern Matcher will log a warning
- Pattern matching will continue using LLM only (without RAG)
- No errors will be raised - the system gracefully degrades

### Performance Tips

- **Index Size**: Keep the index size reasonable (< 100K documents)
- **Chunk Size**: Adjust `chunk_size` based on your pattern document length
- **Retrieval**: The system uses MMR (Maximal Marginal Relevance) search with `k=10` and `fetch_k=30`
- **Embedding Model**: `text-embedding-ada-002` provides good balance of speed and quality

### Troubleshooting

**Q: "FAISS index not found" warning**
- Check that `soc_automation/data/faiss_index/` contains `index.faiss` and `index.pkl`
- Verify the path is correct relative to the project root

**Q: "Failed to initialize RAG components" error**
- Check OPENAI_API_KEY is set correctly
- Verify OPENAI_BASE_URL if using custom endpoint
- Ensure network connectivity to API endpoint

**Q: "Error retrieving context" in logs**
- FAISS index may be corrupted - try recreating it
- Check embeddings model compatibility
- Verify `allow_dangerous_deserialization=True` is set when loading

## Example Error Pattern Documents

Store your error pattern documents in a structured format:

```markdown
# MEM-001: Cache Coherency Violation

## Pattern
- Keywords: cache coherency, coherence violation, snoop conflict
- Base Severity: 9

## Description
Cache coherency violation occurs when multiple caches have inconsistent
copies of the same memory location.

## Common Causes
- MESI protocol state machine error
- Snoop logic bug in cache controller
- Multi-core synchronization issue

## Resolution Steps
1. Review cache coherency protocol implementation
2. Check snoop logic in cache controller
3. Verify MESI state transitions
```

## Example SOP Documents

Store your SOP (Standard Operating Procedure) documents alongside error patterns:

```markdown
# SOP-TIM-001: Simulation Timeout Resolution

## Problem
Simulation timeout errors occur when the simulation exceeds the allocated time limit.

## Resolution Steps
1. Check for infinite loops in the testbench or DUT
2. Review clock gating logic for stuck signals
3. Increase simulation timeout in configuration file
   - Open `sim.cfg`
   - Update: `timeout = 7200`  (from 3600)
4. Rerun the simulation
5. Verify simulation completes successfully

## Prerequisites
- Access to simulation configuration files
- Understanding of testbench structure
- Review privileges for RTL code

## Warnings
- Do not set timeout too high - may mask real design issues
- Always investigate root cause before increasing timeout
- Review simulation logs after timeout change

## Automation
Steps 3-5 can be automated safely.
Manual review required for steps 1-2.
```

## Example Past Solution Documents

```markdown
# Solution: Cache Coherency Fix - Issue #1234

## Problem
Cache coherency violations in L2 cache during multi-core stress tests.
Pattern: MEM-001

## Root Cause
Race condition in cache controller snoop logic when handling
simultaneous requests from multiple cores.

## Solution Applied
Modified `cache_controller.sv` lines 234-267:
- Added mutex for snoop request handling
- Implemented priority arbitration
- Added assertion for conflict detection

## Results
- Issue resolved in build #5678
- Verified with 10,000 regression runs
- No recurrence in 6 months

## Related
- SOP-MEM-001 was updated based on this fix
- New assertion SVA-MEM-012 added to catch early
```

The FAISS index will use all these documents (patterns + SOPs + solutions) to provide comprehensive context during error analysis.
