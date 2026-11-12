# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=2.3", "chromadb>=0.5", "ollama>=0.4"]
# ///
"""Image MCP: 사진을 올리면 설명문을 만들어 벡터로 저장하고, 말로 검색한다.

흐름: 이미지 → Vision 모델이 설명문 작성 → 설명문을 Embedding → Chroma에 저장
검색: 질문을 Embedding → 가까운 설명문의 이미지를 돌려줌 → show_image로 실제 사진 확인

실습에는 공개 데이터셋만 쓴다. 실제 환자 사진은 넣지 않는다.
"""
import hashlib
import os
from pathlib import Path

import chromadb
import ollama
from mcp.server.mcpserver import Image, MCPServer

EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
VISION_MODEL = os.getenv("VISION_MODEL", "gemma3:4b")
DB_DIR = os.path.expanduser(os.getenv("IMAGE_DB", "~/.lab-mcp/images"))
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DESCRIBE_PROMPT = (
    "이 이미지를 나중에 검색할 수 있도록 한국어로 설명해 줘. "
    "어떤 종류의 사진인지, 신체 부위나 대상, 눈에 띄는 특징을 3~5문장으로 써. "
    "진단을 내리지 말고 보이는 것만 써."
)

server = MCPServer("images")
col = chromadb.PersistentClient(path=DB_DIR).get_or_create_collection(
    "images", metadata={"hnsw:space": "cosine"}
)


def embed(text: str) -> list[float]:
    return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]


def describe(path: Path) -> str:
    """Vision 모델에게 이미지 설명을 받는다."""
    res = ollama.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": DESCRIBE_PROMPT, "images": [str(path)]}],
    )
    return res["message"]["content"].strip()


def resolve(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"파일을 찾을 수 없습니다: {p}")
    if p.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"지원하지 않는 형식입니다: {p.suffix} (png, jpg, jpeg, webp만 가능)")
    return p


@server.tool()
def add_image(path: str, note: str = "") -> dict:
    """이미지 한 장을 등록한다. note에는 사람이 붙이는 메모(예: "공개 데이터셋, 폐렴 라벨")를 적는다."""
    p = resolve(path)
    image_id = hashlib.sha1(p.read_bytes()).hexdigest()[:12]
    description = describe(p)
    text = f"{note}\n{description}".strip()
    col.upsert(
        ids=[image_id],
        documents=[text],
        embeddings=[embed(text)],
        metadatas=[{"path": str(p), "note": note}],
    )
    return {"id": image_id, "path": str(p), "description": description}


@server.tool()
def add_folder(folder: str, note: str = "") -> dict:
    """폴더 안의 이미지를 모두 등록한다. 장수가 많으면 시간이 걸린다."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"폴더를 찾을 수 없습니다: {root}")
    added = [add_image(str(p), note)["id"] for p in sorted(root.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return {"folder": str(root), "added": len(added)}


@server.tool()
def search_images(query: str, k: int = 5) -> list[dict]:
    """말로 이미지를 찾는다. 예: "왼쪽 폐 아래쪽이 하얗게 보이는 흉부 엑스레이"."""
    if col.count() == 0:
        return []
    r = col.query(query_embeddings=[embed(query)], n_results=min(k, col.count()))
    return [
        {"id": i, "path": m["path"], "note": m["note"], "description": d, "distance": round(dist, 3)}
        for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]


@server.tool()
def show_image(image_id: str) -> Image:
    """검색 결과의 id로 실제 이미지를 보여 준다."""
    r = col.get(ids=[image_id])
    if not r["ids"]:
        raise ValueError(f"등록되지 않은 id입니다: {image_id}")
    return Image(path=r["metadatas"][0]["path"])


if __name__ == "__main__":
    server.run()
