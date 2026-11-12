# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=2.3", "chromadb>=0.5", "ollama>=0.4"]
# ///
"""Memory MCP: 기억해 달라는 내용을 저장하고, 의미로 찾고, 기간별로 꺼낸다.

요약은 서버가 하지 않는다. list_memories로 기간의 기억을 넘겨주면
대화 중인 LLM(Claude 등)이 읽고 요약한다.
"""
import os
import uuid
from datetime import datetime, timedelta

import chromadb
import ollama
from mcp.server.mcpserver import MCPServer

EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
DB_DIR = os.path.expanduser(os.getenv("MEMORY_DB", "~/.lab-mcp/memory"))

server = MCPServer("memory")
col = chromadb.PersistentClient(path=DB_DIR).get_or_create_collection(
    "memories", metadata={"hnsw:space": "cosine"}
)


def embed(text: str) -> list[float]:
    """Ollama 임베딩 모델로 문장을 벡터로 바꾼다."""
    return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]


def to_item(mid, doc, meta, distance=None) -> dict:
    item = {"id": mid, "content": doc, "created": meta["created"], "tags": meta["tags"]}
    if distance is not None:
        item["distance"] = round(distance, 3)
    return item


@server.tool()
def remember(content: str, tags: str = "") -> str:
    """사용자가 기억해 달라고 한 내용을 저장한다. tags는 쉼표로 구분한 분류(예: "수업,과제")."""
    now = datetime.now()
    mid = uuid.uuid4().hex[:8]
    col.add(
        ids=[mid],
        documents=[content],
        embeddings=[embed(content)],
        metadatas=[{"tags": tags, "created": now.isoformat(timespec="seconds"), "ts": now.timestamp()}],
    )
    return f"기억했습니다 (id={mid})"


@server.tool()
def recall(query: str, k: int = 5) -> list[dict]:
    """질문과 의미가 가까운 기억을 찾는다. distance가 작을수록 가깝다."""
    if col.count() == 0:
        return []
    r = col.query(query_embeddings=[embed(query)], n_results=min(k, col.count()))
    return [
        to_item(i, d, m, dist)
        for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]


@server.tool()
def list_memories(days: int = 30) -> list[dict]:
    """최근 days일 동안의 기억을 시간순으로 돌려준다. "한 달 요약해줘"에 쓴다."""
    since = (datetime.now() - timedelta(days=days)).timestamp()
    r = col.get(where={"ts": {"$gte": since}})
    items = [to_item(i, d, m) for i, d, m in zip(r["ids"], r["documents"], r["metadatas"])]
    return sorted(items, key=lambda x: x["created"])


@server.tool()
def forget(memory_id: str) -> str:
    """id로 기억 하나를 지운다."""
    col.delete(ids=[memory_id])
    return f"삭제했습니다 (id={memory_id})"


if __name__ == "__main__":
    server.run()
