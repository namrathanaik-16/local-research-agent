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

MODEL = "qwen3:8b"   # match whatever `ollama list` shows
REPORTS_DIR = Path("reports")
MAX_HISTORY_TURNS = 5   # how many past question/answer pairs to remember
CHUNK_CHARS = 4000      # how much text the model sees per read
MAX_CLAIMS = 3          # how many claims to check when verifying
HEADERS = {"User-Agent": "Mozilla/5.0"}

ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")

# ---------- Tools ----------

def web_search(query: str) -> str:
    """Search the web and return the top results with titles, URLs and short snippets.

    Args:
        query: The search query.
    """
    results = DDGS().text(query, max_results=5)
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)

def news_search(query: str) -> str:
    """Search recent news articles. Returns headline, source, date, URL and a short snippet.

    Args:
        query: The news topic to search for.
    """
    results = DDGS().news(query, max_results=6)
    if not results:
        return "No news found."
    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('source', '')} | {str(r.get('date', ''))[:10]}\n"
        f"{r.get('url', '')}\n{r.get('body', '')}"
        for r in results
    )

def search_arxiv(query: str) -> str:
    """Search arXiv for research papers. Returns titles, authors, arXiv IDs and abstracts.

    Args:
        query: Keywords describing the papers to find.
    """
    resp = requests.get(
        "https://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "max_results": 5, "sortBy": "relevance"},
        headers=HEADERS,
        timeout=20,
    )
    root = ET.fromstring(resp.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries = root.findall("a:entry", ns)
    if not entries:
        return "No papers found."
    out = []
    for e in entries:
        title = " ".join(e.findtext("a:title", "", ns).split())
        abstract = " ".join(e.findtext("a:summary", "", ns).split())[:500]
        authors = ", ".join(a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)[:4])
        url = e.findtext("a:id", "", ns)
        year = e.findtext("a:published", "", ns)[:4]
        out.append(f"{title} ({year})\nAuthors: {authors}\nURL: {url}\nAbstract: {abstract}")
    return "\n\n".join(out)

def _to_pdf_url(source: str) -> str:
    """Turn an arXiv ID or arxiv.org link into a direct PDF URL; leave other URLs alone."""
    source = source.strip()
    is_arxiv = "arxiv.org" in source or ARXIV_ID.fullmatch(source)
    match = ARXIV_ID.search(source)
    if is_arxiv and match:
        return f"https://arxiv.org/pdf/{match.group(1)}{match.group(2) or ''}"
    return source

def _pdf_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    reader = PdfReader(io.BytesIO(resp.content))
    pages = [(page.extract_text() or "") for page in reader.pages[:40]]  # cap at 40 pages
    return " ".join(" ".join(pages).split())

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
    "news_search": news_search,
    "search_arxiv": search_arxiv,
    "read_pdf": read_pdf,
    "read_webpage": read_webpage,
}

# ---------- Modes ----------

BASE_PROMPT = """You are a research assistant having an ongoing conversation with the user.
Today's date is {today}.
If a question is a follow-up that can be answered from the earlier conversation, answer directly without searching.
Base your answer ONLY on what you read, and end with a 'Sources' list of the URLs you used.
If sources disagree or you can't find something, say so. Never invent facts, names, numbers, paper titles or URLs."""

MODES = {
    "general": {
        "description": "General web research (default)",
        "tools": ["web_search", "search_arxiv", "read_pdf", "read_webpage"],
        "prompt": """Workflow: search (use search_arxiv for scientific papers, web_search for everything else, more than one query if needed), then read the 2-3 most relevant sources with read_pdf or read_webpage, then write a clear, concise summary.""",
        "footer": "",
    },
    "ai-papers": {
        "description": "AI/ML research papers: finds, reads and summarizes papers",
        "tools": ["search_arxiv", "read_pdf", "web_search", "read_webpage"],
        "prompt": """You specialize in AI and machine learning research.
Workflow: use search_arxiv first to find papers. Read the most relevant 1-2 with read_pdf (arXiv IDs work; use part=2 for methods or results if needed). Use web_search only for context such as blog posts or benchmarks.
Format each paper you cover as:
**Title (year)** - authors
- Problem: what it tries to solve
- Method: the core idea in plain language
- Results: the main numbers or findings
- Limitations: what it does not solve
Say clearly when you only read the abstract and not the full paper.""",
        "footer": "",
    },
    "news": {
        "description": "Current events: recent articles, dated and sourced",
        "tools": ["news_search", "web_search", "read_webpage"],
        "prompt": """You specialize in current events.
Workflow: use news_search first, and try 2 different queries. Read the 2-3 most relevant articles with read_webpage, preferring different outlets so you can compare how they report the story.
Format: a one-sentence headline summary, then 3-5 bullet points with the key facts. Put the publication and date next to each fact. Add a short 'What is still unclear' line if reports differ or details are missing.
Keep facts separate from opinion, and label opinion as opinion.""",
        "footer": "",
    },
    "finance": {
        "description": "Companies and markets: filings, results and news (information only)",
        "tools": ["news_search", "web_search", "read_webpage", "read_pdf"],
        "prompt": """You specialize in finance and business research.
Workflow: use web_search for company filings, investor relations pages and earnings releases, and news_search for recent developments. Prefer primary sources (company filings, investor relations, regulators such as the SEC) over commentary. Read the 2-3 most relevant sources (read_pdf for annual reports and filings).
Format: a short overview, then key figures (revenue, profit, growth, guidance) with the period and source for every number, then recent developments, then risks.
Never quote a number you did not see in a source. Never give buy, sell or hold recommendations or price predictions. Present facts and let the user decide.""",
        "footer": "\n\n---\n*Information only, not financial advice. Check figures against the original filings.*",
    },
}

def build_system_prompt(mode: str) -> str:
    base = BASE_PROMPT.format(today=datetime.now().strftime("%B %d, %Y"))
    return base + "\n\n" + MODES[mode]["prompt"]

def tools_for(mode: str) -> list:
    return [TOOLS[name] for name in MODES[mode]["tools"]]

# ---------- Reports ----------

def save_report(question: str, answer: str, mode: str) -> Path:
    """Save the answer as a Markdown file in the reports folder."""
    REPORTS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:50] or "report"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = REPORTS_DIR / f"{stamp}_{mode}_{slug}.md"
    content = (
        f"# {question}\n\n"
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} with {MODEL} | Mode: {mode}*\n\n"
        f"{answer}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path

def strip_thinking(text: str) -> str:
    """Remove any leaked <think> reasoning from the model's reply."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()

# ---------- Verification ----------

def ask_model(prompt: str) -> str:
    """One plain model call with no tools."""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 8192},
        think=False,
    )
    return strip_thinking(response.message.content)

def extract_claims(answer: str) -> list:
    """Ask the model for the key factual claims in an answer."""
    prompt = (
        f"Here is a research answer:\n\n{answer}\n\n"
        f"List the {MAX_CLAIMS} most important factual claims it makes. "
        "Each claim must be one short, self-contained sentence that names its subject "
        "(no pronouns like 'it' or 'this'). Output one claim per line and nothing else."
    )
    lines = ask_model(prompt).splitlines()
    claims = []
    for line in lines:
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip()
        if len(line) > 15:
            claims.append(line)
    return claims[:MAX_CLAIMS]

def check_claim(claim: str, cited_urls: set) -> tuple:
    """Search for independent evidence on one claim. Returns (verdict, reason, url)."""
    results = DDGS().text(claim[:120], max_results=6) or []
    # a "second source" must not be one the answer already used
    fresh = [r for r in results if r["href"].rstrip("/") not in cited_urls][:4]
    if not fresh:
        return "UNCLEAR", "No independent source found.", ""
    evidence = "\n\n".join(f"[{i + 1}] {r['href']}\n{r['body']}" for i, r in enumerate(fresh))
    prompt = (
        f"Claim: {claim}\n\nEvidence from other sources:\n{evidence}\n\n"
        "Do these sources support the claim? Answer in exactly this format:\n"
        "VERDICT: SUPPORTED or CONTRADICTED or UNCLEAR\n"
        "SOURCE: the number of the most relevant source, or 0 if none\n"
        "REASON: one short sentence"
    )
    reply = ask_model(prompt)
    verdict = re.search(r"VERDICT:\s*(SUPPORTED|CONTRADICTED|UNCLEAR)", reply, re.I)
    source = re.search(r"SOURCE:\s*(\d+)", reply)
    reason = re.search(r"REASON:\s*(.+)", reply)
    verdict = verdict.group(1).upper() if verdict else "UNCLEAR"
    idx = int(source.group(1)) if source else 0
    url = fresh[idx - 1]["href"] if 1 <= idx <= len(fresh) else ""
    return verdict, (reason.group(1).strip() if reason else "No clear reason given."), url

def verify_answer(answer: str) -> str:
    """Check the key claims of an answer against independent sources."""
    cited = {u.rstrip("/").rstrip(").,") for u in re.findall(r"https?://[^\s)\]]+", answer)}
    claims = extract_claims(answer)
    if not claims:
        return "Could not extract claims to verify."
    labels = {
        "SUPPORTED": "\u2705 Confirmed by another source",
        "UNCLEAR": "\u26a0\ufe0f Not confirmed (unclear or no other source)",
        "CONTRADICTED": "\u274c Other sources disagree",
    }
    lines = []
    for claim in claims:
        print(f"  -> checking: {claim}")
        try:
            verdict, reason, url = check_claim(claim, cited)
        except Exception as e:
            verdict, reason, url = "UNCLEAR", f"Check failed: {e}", ""
        entry = f"- **{claim}**\n  {labels[verdict]}. {reason}"
        if url:
            entry += f"\n  Source: {url}"
        lines.append(entry)
    return "\n\n".join(lines)

# ---------- Agent loop ----------

def run_agent(question: str, history: list, mode: str = "general", max_steps: int = 10) -> str:
    """Answer one question in the given mode. `history` holds earlier Q&A pairs only."""
    messages = (
        [{"role": "system", "content": build_system_prompt(mode)}]
        + history
        + [{"role": "user", "content": question}]
    )
    tools = tools_for(mode)
    allowed = set(MODES[mode]["tools"])
    for _ in range(max_steps):
        print("  ... thinking")
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=tools,
            options={"num_ctx": 8192},
            think=False,
        )
        msg = response.message
        messages.append(msg)
        if not msg.tool_calls:
            return strip_thinking(msg.content) + MODES[mode]["footer"]
        for call in msg.tool_calls:
            name, args = call.function.name, call.function.arguments
            print(f"  -> using {name}({args})")
            try:
                if name not in allowed:
                    raise ValueError(f"{name} is not available in {mode} mode")
                result = TOOLS[name](**args)
            except Exception as e:
                result = f"Tool error: {e}"
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})
    return "Stopped after too many steps."

def show_modes(current: str) -> None:
    print("Modes:")
    for name, m in MODES.items():
        marker = "*" if name == current else " "
        print(f" {marker} {name:<10} {m['description']}")
    print("Switch with: mode ai-papers | mode news | mode finance | mode general")

if __name__ == "__main__":
    history = []        # remembered question/answer pairs
    last_answer = None  # most recent answer, for 'verify'
    last_path = None    # its saved report file
    mode = "general"
    print("Commands: 'mode' lists modes, 'verify' checks the last answer, 'new' starts fresh, 'quit' exits.")
    show_modes(mode)
    while True:
        q = input(f"\n[{mode}] Research question (or 'mode' / 'verify' / 'new' / 'quit'): ").strip()
        low = q.lower()
        if low in ("quit", "exit", ""):
            break
        if low == "new":
            history, last_answer, last_path = [], None, None
            print("[Memory cleared. Starting a new conversation.]")
            continue
        if low == "mode" or low.startswith("mode "):
            choice = low[4:].strip()
            if choice in MODES:
                mode = choice
                print(f"[Switched to {mode} mode: {MODES[mode]['description']}]")
            else:
                show_modes(mode)
            continue
        if low == "verify":
            if not last_answer:
                print("[Nothing to verify yet. Ask a question first.]")
                continue
            print("  Verifying key claims against other sources...")
            report = verify_answer(last_answer)
            print("\n--- Verification ---\n" + report)
            with open(last_path, "a", encoding="utf-8") as f:
                f.write("\n\n## Verification\n\n" + report + "\n")
            print(f"\n[Added to {last_path}]")
            continue
        answer = run_agent(q, history, mode)
        print("\n" + answer)
        if not answer.startswith("Stopped after"):
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": answer})
            history = history[-MAX_HISTORY_TURNS * 2:]
            last_answer = answer
            last_path = save_report(q, answer, mode)
            print(f"\n[Saved to {last_path}]")
            print("[Type 'verify' to fact-check this answer.]")