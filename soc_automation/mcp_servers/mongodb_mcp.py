"""
MongoDB MCP Server using FastMCP.

FastMCP 기반 MongoDB MCP 서버
- Resource: 데이터베이스/컬렉션/문서 목록
- Tool: find, insert_one, aggregate (읽기 전용)
"""

import os
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("MongoDB MCP Server")

# Global MongoDB client
_mongo_client: Optional[AsyncIOMotorClient] = None


def get_mongo_client() -> AsyncIOMotorClient:
    """Get or create MongoDB client."""
    global _mongo_client
    if _mongo_client is None:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        _mongo_client = AsyncIOMotorClient(uri)
    return _mongo_client


def serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert BSON document to JSON-serializable format."""
    if doc is None:
        return None

    result = {}
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            result[key] = str(value)
        elif isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, dict):
            result[key] = serialize_doc(value)
        elif isinstance(value, list):
            result[key] = [serialize_doc(v) if isinstance(v, dict) else v for v in value]
        else:
            result[key] = value
    return result


# ==================== Resources ====================

@mcp.resource("mongodb://databases")
async def list_databases() -> str:
    """List all MongoDB databases."""
    try:
        client = get_mongo_client()
        db_list = await client.list_database_names()
        return json.dumps({"databases": db_list}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to list databases: {str(e)}"})


@mcp.resource("mongodb://{db_name}/collections")
async def list_collections(db_name: str) -> str:
    """List all collections in a database."""
    try:
        client = get_mongo_client()
        db = client[db_name]
        collections = await db.list_collection_names()
        return json.dumps({
            "database": db_name,
            "collections": collections
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to list collections: {str(e)}"})


@mcp.resource("mongodb://{db_name}/{collection_name}/{doc_id}")
async def get_document(db_name: str, collection_name: str, doc_id: str) -> str:
    """Get a specific document by ID."""
    try:
        client = get_mongo_client()
        collection = client[db_name][collection_name]

        # Try to convert to ObjectId, fallback to string
        try:
            query = {"_id": ObjectId(doc_id)}
        except Exception:
            query = {"_id": doc_id}

        doc = await collection.find_one(query)
        if doc is None:
            return json.dumps({
                "error": f"Document not found: {doc_id}",
                "database": db_name,
                "collection": collection_name
            })

        return json.dumps({
            "database": db_name,
            "collection": collection_name,
            "document": serialize_doc(doc)
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get document: {str(e)}"})


# ==================== Tools ====================

@mcp.tool()
async def find(
    database: str,
    collection: str,
    filter: Optional[Dict[str, Any]] = None,
    limit: int = 10,
    projection: Optional[Dict[str, Any]] = None
) -> str:
    """
    Find documents in MongoDB collection.

    Args:
        database: Database name
        collection: Collection name
        filter: Query filter (default: {})
        limit: Maximum number of documents to return (default: 10, max: 100)
        projection: Fields to include/exclude

    Returns:
        JSON string with matching documents
    """
    try:
        client = get_mongo_client()
        coll = client[database][collection]

        # Validate and limit
        if filter is None:
            filter = {}
        if limit > 100:
            limit = 100

        # Convert string ObjectIds in filter
        if "_id" in filter and isinstance(filter["_id"], str):
            try:
                filter["_id"] = ObjectId(filter["_id"])
            except Exception:
                pass

        # Execute query
        cursor = coll.find(filter, projection).limit(limit)
        docs = await cursor.to_list(length=limit)

        return json.dumps({
            "database": database,
            "collection": collection,
            "count": len(docs),
            "documents": [serialize_doc(doc) for doc in docs]
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Find operation failed: {str(e)}",
            "database": database,
            "collection": collection
        })


@mcp.tool()
async def insert_one(
    database: str,
    collection: str,
    document: Dict[str, Any]
) -> str:
    """
    Insert a single document into MongoDB collection.

    Args:
        database: Database name
        collection: Collection name
        document: Document to insert

    Returns:
        JSON string with inserted document ID
    """
    try:
        client = get_mongo_client()
        coll = client[database][collection]

        # Insert document
        result = await coll.insert_one(document)

        return json.dumps({
            "success": True,
            "database": database,
            "collection": collection,
            "inserted_id": str(result.inserted_id)
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Insert operation failed: {str(e)}",
            "database": database,
            "collection": collection
        })


@mcp.tool()
async def aggregate(
    database: str,
    collection: str,
    pipeline: List[Dict[str, Any]]
) -> str:
    """
    Execute read-only aggregation pipeline.

    Args:
        database: Database name
        collection: Collection name
        pipeline: Aggregation pipeline stages

    Returns:
        JSON string with aggregation results
    """
    try:
        # Block write operations
        blocked_stages = ["$out", "$merge"]
        for stage in pipeline:
            for key in stage.keys():
                if key in blocked_stages:
                    return json.dumps({
                        "error": f"Write operation '{key}' is not allowed in aggregation",
                        "blocked_stages": blocked_stages
                    })

        client = get_mongo_client()
        coll = client[database][collection]

        # Execute aggregation
        cursor = coll.aggregate(pipeline)
        results = await cursor.to_list(length=100)

        return json.dumps({
            "database": database,
            "collection": collection,
            "count": len(results),
            "results": [serialize_doc(doc) for doc in results]
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Aggregation failed: {str(e)}",
            "database": database,
            "collection": collection
        })


# ==================== Server Entry Point ====================

if __name__ == "__main__":
    # Run FastMCP server
    mcp.run()
