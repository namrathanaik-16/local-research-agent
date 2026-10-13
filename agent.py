import re
from datetime import datetime
from pathlib import Path

import ollama
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

MODEL = "qwen3:4b"   # match whatever `ollama list` shows
REPORTS_DIR = Path("reports")
MAX_HISTORY_TURNS = 5   # how many past question/answer pairs to remember

def web_search(query: str) -> str:
    """Search the web and return the top results with titles, URLs and short snippets.

    Args:
        query: The search query.
    """
    results = DDGS().text(query, max_results=5)
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)

def read_webpage(url: str) -> str:
    """Fetch a web page and return its main text content.

    Args:
        url: The full URL of the page to read.
    """
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:25 00]

TOOLS = {"web_search": web_search, "read_webpage": read_webpage}

SYSTEM_PROMPT = """You are a research assistant having an ongoing conversation with the user.
For each new question:
1. If it is a follow-up that can be answered from the earlier conversation, answer directly without searching.
2. Otherwise, search the web, using more than one query if needed.
3. Read the 2-3 most relevant pages with read_webpage.
4. Write a clear, concise summary based ONLY on what you read.
5. End with a 'Sources' list of the URLs you used.
If sources disagree or you can't find something, say so. Never invent facts or URLs."""

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