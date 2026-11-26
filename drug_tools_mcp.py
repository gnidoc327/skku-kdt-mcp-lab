"""Drug Review MCP: 신약 후보물질 검토에 쓰는 공개 데이터베이스 조회 도구를 MCP로 내놓는다.

Kiro나 Claude에 연결하면, 그 AI가 지휘자(Orchestrator)가 되어 도구를 골라 부르고
통합 리포트를 쓴다. 노트북 2-1, 2-2에서 쓴 drug_tools.py를 그대로 감싼 것이다.
교육·연구용이며 임상 판단을 대신하지 않는다.
"""
from mcp.server.mcpserver import MCPServer

import drug_tools as dt

server = MCPServer("drug-review")


@server.tool()
def compound_info(name: str) -> dict:
    """[화학] 화합물의 영문 이름(예: aspirin)으로 PubChem의 기본 정보와 SMILES 구조식을 찾는다. 검토의 첫 단계로 부른다."""
    return dt.compound_info(name)


@server.tool()
def molecule_properties(smiles: str) -> dict:
    """[화학] SMILES 구조식으로 분자량, 지용성(logP), 수소결합 수, TPSA, QED, Lipinski 규칙 통과 여부를 계산한다."""
    return dt.molecule_properties(smiles)


@server.tool()
def similar_compounds(smiles: str, limit: int = 5) -> dict:
    """[화학] SMILES 구조식과 구조가 비슷한 화합물을 PubChem에서 찾는다."""
    return dt.similar_compounds(smiles, limit)


@server.tool()
def pubmed_search(query: str, limit: int = 5) -> dict:
    """[임상약리] PubMed에서 논문을 검색한다. 검색어는 영문으로 쓴다 (예: "ibuprofen renal toxicity")."""
    return dt.pubmed_search(query, limit)


@server.tool()
def adverse_events(drug: str, limit: int = 8) -> dict:
    """[임상약리] openFDA에 가장 많이 보고된 부작용. 영문 성분명을 넣는다. 보고 건수이지 발생률이 아니다."""
    return dt.adverse_events(drug, limit)


@server.tool()
def clinical_trials(query: str, limit: int = 5) -> dict:
    """[임상·규제] ClinicalTrials.gov에서 임상시험을 검색한다. 약 이름이나 질환을 영문으로 넣는다."""
    return dt.clinical_trials(query, limit)


if __name__ == "__main__":
    server.run()
