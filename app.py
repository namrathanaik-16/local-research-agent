import contextlib
import io

import streamlit as st

import agent

st.set_page_config(page_title="Local Research Agent", page_icon="🔎", layout="centered")

# ---------- Session state (Streamlit reruns this script on every click) ----------

def init_state():
    st.session_state.setdefault("history", [])    # Q&A pairs the model remembers
    st.session_state.setdefault("messages", [])   # everything shown on screen

init_state()

# ---------- Sidebar ----------

with st.sidebar:
    st.title("🔎 Research Agent")
    st.caption(f"Model: {agent.MODEL}")

    mode = st.selectbox(
        "Mode",
        list(agent.MODES.keys()),
        format_func=lambda m: f"{m}: {agent.MODES[m]['description']}",
    )

    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Saved reports")
    report_files = []
    if agent.REPORTS_DIR.exists():
        report_files = sorted(agent.REPORTS_DIR.glob("*.md"), reverse=True)
    chosen = st.selectbox(
        "Open a report",
        [None] + report_files,
        format_func=lambda p: "Choose..." if p is None else p.name,
    )

# ---------- Main page ----------

st.title("Local Research Agent")
st.caption("Runs on your own machine with an open-source model. Answers can contain mistakes, so check the sources.")

if chosen is not None:
    text = chosen.read_text(encoding="utf-8")
    with st.expander(f"📄 {chosen.name}", expanded=True):
        st.markdown(text)
        st.download_button("Download", text, file_name=chosen.name, mime="text/markdown")

# Show the conversation so far
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] != "assistant":
            continue
        if m.get("steps"):
            with st.expander("Steps the agent took"):
                st.code("\n".join(m["steps"]))
        if m.get("path"):
            st.caption(f"Saved to {m['path']}")
        if m.get("verification"):
            st.markdown("**Verification**")
            st.markdown(m["verification"])
        elif m.get("path") and st.button("✅ Verify this answer", key=f"verify_{i}"):
            with st.spinner("Checking key claims against other sources..."):
                try:
                    report = agent.verify_answer(m["content"])
                    with open(m["path"], "a", encoding="utf-8") as f:
                        f.write("\n\n## Verification\n\n" + report + "\n")
                    m["verification"] = report
                except Exception as e:
                    m["verification"] = f"Verification failed: {e}"
            st.rerun()

# New question
question = st.chat_input(f"Ask a research question ({mode} mode)")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        log = io.StringIO()
        try:
            with st.spinner("Researching... local models can take a minute or two"):
                with contextlib.redirect_stdout(log):
                    answer = agent.run_agent(question, st.session_state.history, mode)
        except Exception as e:
            answer = f"Something went wrong: {e}\n\nIs Ollama running, and does MODEL in agent.py match `ollama list`?"

    steps = [ln.strip() for ln in log.getvalue().splitlines() if "->" in ln]
    entry = {"role": "assistant", "content": answer, "steps": steps}

    if not answer.startswith(("Stopped after", "Something went wrong")):
        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "assistant", "content": answer})
        st.session_state.history = st.session_state.history[-agent.MAX_HISTORY_TURNS * 2:]
        entry["path"] = str(agent.save_report(question, answer, mode))

    st.session_state.messages.append(entry)
    st.rerun()