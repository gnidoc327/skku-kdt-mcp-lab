"""신약 후보물질 검토용 도구 모음. 공개 데이터베이스(키 필요 없음)를 조회한다.

- compound_info      : PubChem에서 화합물 기본 정보와 SMILES
- molecule_properties: RDKit으로 물리화학적 특성과 Lipinski 규칙 계산 (내 컴퓨터에서 계산)
- similar_compounds  : PubChem에서 구조가 비슷한 화합물
- pubmed_search      : PubMed 논문 검색 (제목, 연도, 학술지)
- clinical_trials    : ClinicalTrials.gov 임상시험 검색
- adverse_events     : openFDA에 보고된 부작용 상위 항목

같은 요청은 data/cache 폴더에 저장해 두고 다시 쓴다. 수업처럼 여러 명이 동시에
조회할 때 공개 API의 요청 한도에 걸리지 않게 하기 위해서다.
교육·연구용이며 임상 판단을 대신하지 않는다.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import quote

import requests

CACHE_DIR = Path(os.getenv("DRUG_CACHE", Path(__file__).resolve().parent / "data" / "cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TIMEOUT = 20
HEADERS = {"User-Agent": "mcp-lab-classroom/1.0"}


def _get(url: str, params: dict | None = None) -> dict:
    """GET 요청. 같은 요청은 파일로 저장해 두고 다시 쓴다."""
    key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()[:20]
    f = CACHE_DIR / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT, headers=HEADERS)
            if r.status_code == 404:
                data = {"_not_found": True}
                break
            if r.status_code in (429, 503):       # 요청이 몰리면 잠깐 쉬고 다시
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:  # 네트워크 오류는 도구 결과로 돌려준다
            last = e
            time.sleep(1.0)
    else:
        return {"_error": f"조회 실패: {last or '요청 한도 초과'}"}
    f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPS = "MolecularFormula,MolecularWeight,ConnectivitySMILES,SMILES,IUPACName,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount"


def compound_info(name: str) -> dict:
    """화합물 이름(영문, 예: aspirin)으로 PubChem의 기본 정보와 SMILES를 찾는다."""
    d = _get(f"{PUBCHEM}/compound/name/{quote(name)}/property/{PROPS}/JSON")
    if "_error" in d or "_not_found" in d or "PropertyTable" not in d:
        return {"found": False, "name": name, "message": d.get("_error", "PubChem에서 찾지 못했습니다. 영문 이름을 확인하세요.")}
    p = d["PropertyTable"]["Properties"][0]
    return {
        "found": True,
        "name": name,
        "cid": p.get("CID"),
        "formula": p.get("MolecularFormula"),
        "molecular_weight": p.get("MolecularWeight"),
        "smiles": p.get("SMILES") or p.get("ConnectivitySMILES"),
        "iupac_name": p.get("IUPACName"),
        "xlogp": p.get("XLogP"),
        "tpsa": p.get("TPSA"),
        "source": f"https://pubchem.ncbi.nlm.nih.gov/compound/{p.get('CID')}",
    }


def molecule_properties(smiles: str) -> dict:
    """SMILES 구조식으로 분자량, 지용성(logP), 수소결합 수 등을 계산하고 Lipinski 규칙을 확인한다."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ImportError:
        return {"valid": False, "message": "rdkit이 설치되지 않았습니다. pip install rdkit"}
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "smiles": smiles, "message": "SMILES를 해석할 수 없습니다."}
    mw, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
    hbd, hba = Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol)
    violations = [
        msg for bad, msg in [
            (mw > 500, "분자량 500 초과"),
            (logp > 5, "logP 5 초과"),
            (hbd > 5, "수소결합 주개 5개 초과"),
            (hba > 10, "수소결합 받개 10개 초과"),
        ] if bad
    ]
    return {
        "valid": True,
        "smiles": smiles,
        "molecular_weight": round(mw, 1),
        "logp": round(logp, 2),
        "h_bond_donors": hbd,
        "h_bond_acceptors": hba,
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 1),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "qed": round(QED.qed(mol), 2),
        "lipinski_violations": violations,
        "lipinski_pass": len(violations) <= 1,
        "note": "QED는 0~1, 클수록 먹는 약에 가까운 성질. Lipinski는 위반 1개까지 통과로 본다.",
    }


def similar_compounds(smiles: str, limit: int = 5) -> dict:
    """PubChem에서 구조가 비슷한 화합물을 찾는다 (2D 유사도 90% 이상)."""
    d = _get(f"{PUBCHEM}/compound/fastsimilarity_2d/smiles/{quote(smiles, safe='')}/property/Title,MolecularFormula,MolecularWeight/JSON",
             {"Threshold": 90, "MaxRecords": limit + 1})
    if "_error" in d or "_not_found" in d or "PropertyTable" not in d:
        return {"count": 0, "items": [], "message": d.get("_error", "비슷한 화합물을 찾지 못했습니다.")}
    items = [{"cid": p.get("CID"), "name": p.get("Title"), "formula": p.get("MolecularFormula"), "molecular_weight": p.get("MolecularWeight")}
             for p in d["PropertyTable"]["Properties"]][:limit + 1]
    return {"count": len(items), "items": items}


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def pubmed_search(query: str, limit: int = 5) -> dict:
    """PubMed에서 논문을 검색한다. 영문 검색어를 쓴다 (예: "aspirin cardiotoxicity")."""
    s = _get(f"{EUTILS}/esearch.fcgi", {"db": "pubmed", "term": query, "retmax": limit, "retmode": "json", "sort": "relevance"})
    if "_error" in s:
        return {"count": 0, "items": [], "message": s["_error"]}
    ids = s.get("esearchresult", {}).get("idlist", [])
    total = int(s.get("esearchresult", {}).get("count", 0))
    if not ids:
        return {"total": total, "count": 0, "items": []}
    d = _get(f"{EUTILS}/esummary.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
    res = d.get("result", {})
    items = [{"pmid": i, "title": res[i].get("title"), "journal": res[i].get("source"), "year": (res[i].get("pubdate") or "")[:4],
              "url": f"https://pubmed.ncbi.nlm.nih.gov/{i}/"} for i in ids if i in res]
    return {"total": total, "count": len(items), "items": items}


def clinical_trials(query: str, limit: int = 5) -> dict:
    """ClinicalTrials.gov에서 임상시험을 검색한다. 약 이름이나 질환을 영문으로 넣는다."""
    d = _get("https://clinicaltrials.gov/api/v2/studies", {"query.term": query, "pageSize": limit, "countTotal": "true"})
    if "_error" in d or "studies" not in d:
        return {"count": 0, "items": [], "message": d.get("_error", "임상시험을 찾지 못했습니다.")}
    items = []
    for st in d["studies"]:
        ps = st.get("protocolSection", {})
        idm, sm, dm = ps.get("identificationModule", {}), ps.get("statusModule", {}), ps.get("designModule", {})
        items.append({
            "nct_id": idm.get("nctId"),
            "title": idm.get("briefTitle"),
            "status": sm.get("overallStatus"),
            "phase": ", ".join(dm.get("phases", []) or []) or "해당 없음",
            "conditions": (ps.get("conditionsModule", {}).get("conditions") or [])[:3],
            "url": f"https://clinicaltrials.gov/study/{idm.get('nctId')}",
        })
    return {"total": d.get("totalCount"), "count": len(items), "items": items}


def adverse_events(drug: str, limit: int = 8) -> dict:
    """openFDA에 보고된 부작용 중 가장 많이 보고된 항목. 보고 건수이지 발생률이 아니다."""
    d = _get("https://api.fda.gov/drug/event.json",
             {"search": f'patient.drug.medicinalproduct:"{drug}"', "count": "patient.reaction.reactionmeddrapt.exact", "limit": limit})
    if "_error" in d or "results" not in d:
        return {"count": 0, "items": [], "message": d.get("_error", "보고된 부작용을 찾지 못했습니다. 영문 성분명을 확인하세요.")}
    return {"drug": drug, "count": len(d["results"]),
            "items": [{"reaction": r["term"], "reports": r["count"]} for r in d["results"]],
            "note": "자발적 보고 건수입니다. 많이 쓰이는 약일수록 건수가 많고, 인과관계가 확인된 것은 아닙니다."}


if __name__ == "__main__":
    c = compound_info("aspirin")
    print(c)
    print(molecule_properties(c["smiles"]))
    print(similar_compounds(c["smiles"], 3))
    print(pubmed_search("aspirin gastrointestinal bleeding", 3))
    print(clinical_trials("aspirin", 3))
    print(adverse_events("aspirin", 5))
