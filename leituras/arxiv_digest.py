# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx>=0.28.1",
# ]
# ///


import sys
import httpx

URL = "https://export.arxiv.org/api/query"

PARAMS = {
    "search_query": 'cat:cs.CV AND ti:anomaly AND abs:"few-shot"',
    "sortBy": "submittedDate",
    "sortOrder": "descending",
    "max_results": 3,
}

def buscar(params: dict) -> str:
    """Consulta arxiv e retorna XML"""
    resp = httpx.get(URL, params=params, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    return resp.text

def main() -> int:
    try:
        xml = buscar(PARAMS)
    except httpx.HTTPStatusError as e:
        print(f"Erro HTTP {e.response.status_code} em {e.request.url}", file=sys.stderr)
        return 1
    except httpx.RequestError as e:
        print(f"Falha na requisição: {e!r}", file=sys.stderr)
        return 1

    print(xml[:300])
    return 0

if __name__ == "__main__":
    sys.exit(main())
