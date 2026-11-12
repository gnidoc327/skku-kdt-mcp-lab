# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=2.3", "chromadb>=0.5", "ollama>=0.4"]
# ///
"""설치 확인: AI 없이 Memory MCP의 저장과 검색이 되는지 본다. 임시 폴더를 써서 내 기억은 건드리지 않는다."""
import logging
import os
import sys
import tempfile

logging.disable(logging.WARNING)  # 요청 로그는 숨기고 결과만 보여 준다

os.environ["MEMORY_DB"] = tempfile.mkdtemp(prefix="lab-mcp-check-")

try:
    import memory_mcp as m

    m.remember("다음 주 화요일까지 RAG 과제 제출", "과제")
    m.remember("병원 주차장은 지하 2층", "생활")
    top = m.recall("숙제 마감이 언제지?", k=1)[0]
except Exception as e:  # Ollama가 꺼져 있거나 모델이 없을 때가 대부분
    print(f"❌ 실패: {e}")
    print("   Ollama 앱이 켜져 있는지, 'ollama pull bge-m3'를 했는지 확인하세요.")
    sys.exit(1)

if "RAG 과제" in top["content"]:
    print(f"✅ 성공: '숙제 마감이 언제지?' → '{top['content']}' (distance {top['distance']})")
else:
    print(f"❌ 검색 결과가 예상과 다릅니다: {top}")
    sys.exit(1)
