import io 
import re
import xml.etree.ElementTree as ET 
from datetime import datetime
from pathlib import Path 

import ollama
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS 
from pypdf import PdfReader

MODEL = "qwen3:4b"
REPORTS_DIR = Path("reports")
MAX_HISTORY_TURNS = 5
CHUNK_CHARS = 4000
HEADERS = {"User-Agent":"Mozilla/5.0"}

ARVIX_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")

def web_search(query:str)->str:
    """Search the web and return the top results with title, URLs and short snippets.
    
    Args:
        query:The search query.
    """
    results= DDGS().text(query, max_results=5)
    if not results:
        return "No results found"
    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)

def search_arxiv(query:str)->str:
    """Search arVix for research papers. Return titles, authors, arXiv IDs and abstracts.
    Args:
        query:Keywords describing the paper to find.
    """
    resp = requests.get(
        "https://export.arxiv.org/api/query",
        params = {"search_query":f"all:{query}","max_results":5, "sortBy":"relevance"},
        headers= HEADERS,
        timeout=20,
    )
    root = ET.fromstring(resp.text)
    ns = {"a":"http://www.w3.org/2005/Atom"}
    entries = root.findall("a:entry",ns)
    if not entries:
        return "No papers found"
    out = []
    for e in entries:
        title=" ".join(e.findtext("a:title","",ns).split())
        abstract=" ".join(e.findtext("a:summary","",ns).split())[:500]
        authors=" ".join(e.findtext("a:name","",ns)for a in e.findall("a:author",ns)[:4])
        url=e.findtext("a:id","",ns)
        year=e.findtext("a:published","",ns)[:4]
        out.append(f"{title}({year})\nAuthors:{authors}\nURL:{url}\nAbstract:{abstract}")
    return "\n\n".join(out)

def _to_pdf_url(source:str)->str:
    """Turn an arXiv ID or arXiv.org link into a direct PDF URL; leave other URLs alone."""
    source=source.strip()
    is_arxiv="arxiv.org" in source or ARXIV_ID.fullmatch(source)
    match = ARXIV_ID.search(source)
    if is_arxiv and match:
        return f"https://arxiv.org/pdf/{match.group(1)}{match.group(2) or ''}"
    return source

def read_pdf(source:str, part:int=1)->str:
    """Read a research paper or PDF. Accepts an arXiv ID (like 1706.03762), an arxiv.org link, or a different PDF URL.
    Long documents are split into parts; call again with part=2, part=3, and so on to read further.
    
    Args:
        source:An arXiv ID, arxiv.org link, or PDF URL.
        part:Which chunk of the document to return, starting at 1.
    """
    text = _pdf_text(_to_pdf_url(source))
    if not text:
        return "Could not extract text(the PDF may be scanned images.)"
    total = max(1,-(-len(text)//CHUNK_CHARS))
    part = max(1,min(part,total))
    start = (part-1)*CHUNK_CHARS
    return f"[Part{part} of {total}]\n{text[start:start + CHUNK_CHARS]}"

def read_pdf(source: str, part: int = 1) -> str:
    """Read a research paper or PDF. Accepts an arXiv ID (like 1706.03762), an arxiv.org link, or a direct PDF URL.
    Long documents are split into parts; call again with part=2, part=3, and so on to read further.

    Args:
        source: An arXiv ID, arxiv.org link, or PDF URL.
        part: Which chunk of the document to return, starting at 1.
    """
    text = _pdf_text(_to_pdf_url(source))
    if not text:
        return "Could not extract text (the PDF may be scanned images)."
    total = max(1, -(-len(text) // CHUNK_CHARS))   # ceiling division
    part = max(1, min(part, total))
    start = (part - 1) * CHUNK_CHARS
    return f"[Part {part} of {total}]\n{text[start:start + CHUNK_CHARS]}"

def read_webpage(url: str) -> str:
    """Fetch a web page and return its main text content. If the link is a PDF, it is read as a PDF.

    Args:
        url: The full URL of the page to read.
    """
    if url.lower().split("?")[0].endswith(".pdf") or "arxiv.org" in url:
        return read_pdf(url)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if "application/pdf" in resp.headers.get("Content-Type", ""):
        return read_pdf(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:CHUNK_CHARS]

TOOLS = {
    "web_search": web_search,
    "search_arxiv": search_arxiv,
    "read_pdf": read_pdf,
    "read_webpage": read_webpage,
}

SYSTEM_PROMPT = """You are a research assistant having an ongoing conversation with the user.
For each new question:
1. If it is a follow-up that can be answered from the earlier conversation, answer directly without searching.
2. Otherwise, search. Use search_arxiv for scientific or technical papers, and web_search for everything else. Use more than one query if needed.
3. Read the 2-3 most relevant sources: read_pdf for papers and PDFs (arXiv IDs work), read_webpage for web pages.
4. Write a clear, concise summary based ONLY on what you read.
5. End with a 'Sources' list of the URLs you used.
If sources disagree or you can't find something, say so. Never invent facts, paper titles or URLs."""

def save_report(question: str, answer: str) -> Path:
    """Save the answer as a Markdown file in the reports folder."""
    REPORTS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:50] or "report"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = REPORTS_DIR / f"{stamp}_{slug}.md"
    content = (
        f"# {question}\n\n"
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} with {MODEL}*\n\n"
        f"{answer}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path

def run_agent(question: str, history: list, max_steps: int = 10) -> str:
    """Answer one question. `history` holds earlier question/answer pairs only."""
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": question}]
    )
    for _ in range(max_steps):
        print("  ... thinking")
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=list(TOOLS.values()),
            options={"num_ctx": 8192},
            think=False,
        )
        msg = response.message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content
        for call in msg.tool_calls:
            name, args = call.function.name, call.function.arguments
            print(f"  -> using {name}({args})")
            try:
                result = TOOLS[name](**args)
            except Exception as e:
                result = f"Tool error: {e}"
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})
    return "Stopped after too many steps."

if __name__ == "__main__":
    history = []   # remembered question/answer pairs
    print("Type 'new' to start a fresh conversation, 'quit' to exit.")
    while True:
        q = input("\nResearch question (or 'new' / 'quit'): ").strip()
        if q.lower() in ("quit", "exit", ""):
            break
        if q.lower() == "new":
            history = []
            print("[Memory cleared. Starting a new conversation.]")
            continue
        answer = run_agent(q, history)
        print("\n" + answer)
        if not answer.startswith("Stopped after"):
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": answer})
            history = history[-MAX_HISTORY_TURNS * 2:]
            path = save_report(q, answer)
            print(f"\n[Saved to {path}]")