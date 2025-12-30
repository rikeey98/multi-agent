"""
Pattern Matcher Sub-Agent.

100+ 회사 패턴 매칭 + RAG (Retrieval-Augmented Generation)
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from soc_automation.utils.logger import get_agent_logger
from soc_automation.config.settings import settings

logger = get_agent_logger("pattern_matcher")

# Set TIKTOKEN cache directory for offline server
os.environ["TIKTOKEN_CACHE_DIR"] = os.path.expanduser("~/.cache/tiktoken")

# Initialize embeddings and vector store
_embeddings = None
_vector_store = None


def _init_rag():
    """Initialize RAG components (embeddings and vector store)."""
    global _embeddings, _vector_store

    if _embeddings is not None and _vector_store is not None:
        return  # Already initialized

    try:
        # Initialize embeddings
        embeddings_kwargs = {
            "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        }
        if settings.openai.base_url:
            embeddings_kwargs["openai_api_base"] = settings.openai.base_url
        if settings.openai.api_key:
            embeddings_kwargs["openai_api_key"] = settings.openai.api_key

        _embeddings = OpenAIEmbeddings(**embeddings_kwargs)

        # Load FAISS vector store
        faiss_index_path = Path(__file__).parent.parent.parent.parent / "data" / "faiss_index"

        if faiss_index_path.exists():
            _vector_store = FAISS.load_local(
                str(faiss_index_path),
                embeddings=_embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"FAISS vector store loaded from: {faiss_index_path}")
        else:
            logger.warning(f"FAISS index not found at: {faiss_index_path}")
            logger.warning("RAG will not be available. Pattern matching will use LLM only.")
            _vector_store = None

    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)
        _embeddings = None
        _vector_store = None


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    if _vector_store is None:
        return "RAG not available - FAISS index not loaded", []

    try:
        retrieved_docs = _vector_store.max_marginal_relevance_search(query, k=10, fetch_k=30)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return f"Error retrieving context: {str(e)}", []


def _load_prompt() -> str:
    """Load pattern matcher prompt from MD file."""
    prompt_file = Path(__file__).parent.parent.parent.parent.parent / \
                  "soc_automation/config/prompts/error_analyzer/sub_agents/pattern_matcher.md"

    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()


async def run_pattern_matcher(
    llm,
    error_message: str,
    log_context: str = ""
) -> Dict[str, Any]:
    """
    Run pattern matcher to match error against 100+ company patterns.

    Uses RAG (Retrieval-Augmented Generation) to search similar error patterns
    from the vector store before matching.

    Args:
        llm: Language model
        error_message: Error message to match
        log_context: Additional log context (optional)

    Returns:
        {
            "matched": bool,
            "pattern_code": str | None,
            "pattern_name": str | None,
            "base_severity": int | None,
            "confidence": float,
            "reasoning": str
        }
    """
    logger.info("Running pattern matcher with RAG...")

    try:
        # Initialize RAG components
        _init_rag()

        # Prepare tools list
        tools = []
        if _vector_store is not None:
            tools = [retrieve_context]
            logger.info("RAG enabled - retrieve_context tool available")
        else:
            logger.warning("RAG not available - using LLM only")

        # Create agent with pattern matcher prompt
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=_load_prompt()
        )

        # Prepare input
        rag_instruction = ""
        if _vector_store is not None:
            rag_instruction = """
STEP 1: First, use the retrieve_context tool to search for similar error patterns in the knowledge base.
Query the vector store with the error message to find relevant historical patterns.

"""

        input_msg = f"""
Match this error message against company error patterns:

ERROR MESSAGE:
{error_message}

ADDITIONAL CONTEXT:
{log_context if log_context else "No additional context"}

{rag_instruction}STEP 2: Analyze the error (using retrieved context if available) and return a JSON object with:
- matched: true if pattern found with confidence > 0.7, false otherwise
- pattern_code: the pattern code (e.g., "MEM-001") or null
- pattern_name: the pattern name or null
- base_severity: the base severity (0-10) or null
- confidence: confidence score (0.0-1.0)
- reasoning: explanation of the match or why no match

Be precise. Only match if you are confident (> 0.7).
If you used retrieve_context, mention which sources helped identify the pattern.
"""

        # Run agent
        result = await agent.ainvoke({"messages": [HumanMessage(content=input_msg)]})

        # Extract result from agent output
        messages = result.get("messages", [])
        response_text = messages[-1].content if messages else "{}"

        # Try to parse JSON from response
        try:
            # Find JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                pattern_result = json.loads(json_str)
            else:
                # Fallback: no match
                pattern_result = {
                    "matched": False,
                    "pattern_code": None,
                    "pattern_name": None,
                    "base_severity": None,
                    "confidence": 0.0,
                    "reasoning": "Failed to parse pattern match result"
                }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from pattern matcher response")
            pattern_result = {
                "matched": False,
                "pattern_code": None,
                "pattern_name": None,
                "base_severity": None,
                "confidence": 0.0,
                "reasoning": f"JSON parse error. Raw response: {response_text[:200]}"
            }

        logger.info(f"Pattern match result: matched={pattern_result.get('matched')}, "
                   f"pattern={pattern_result.get('pattern_code')}")

        return pattern_result

    except Exception as e:
        logger.error(f"Pattern matcher failed: {e}", exc_info=True)
        return {
            "matched": False,
            "pattern_code": None,
            "pattern_name": None,
            "base_severity": None,
            "confidence": 0.0,
            "reasoning": f"Error during pattern matching: {str(e)}"
        }


if __name__ == "__main__":
    """Test pattern matcher independently."""
    import os
    import asyncio
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    print("=== Pattern Matcher Sub-Agent Test ===\n")

    # Test cases
    test_cases = [
        "Cache coherency violation detected at address 0x1000",
        "AXI SLVERR on write transaction to 0x2000",
        "Simulation timeout after 10000000 cycles",
        "Unknown strange error in module XYZ",
    ]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        exit(1)

    llm_kwargs = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0.7
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        llm_kwargs["base_url"] = base_url
    llm = ChatOpenAI(**llm_kwargs)

    async def test():
        for i, error_msg in enumerate(test_cases, 1):
            print(f"\nTest {i}: {error_msg}")
            print("-" * 80)

            result = await run_pattern_matcher(llm, error_msg)

            print(f"Matched: {result['matched']}")
            print(f"Pattern: {result['pattern_code']} - {result['pattern_name']}")
            print(f"Base Severity: {result['base_severity']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Reasoning: {result['reasoning']}")
            print()

    asyncio.run(test())
    print("\nTest completed.")
