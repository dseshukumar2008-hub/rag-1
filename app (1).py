import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🤖")


def resolve_corpus_path():
    """Find the HR corpus folder. Checks an explicit env var first, then
    falls back to searching common locations so this works whether the PDFs
    live in ./hr_corpus/, a Kaggle-style nested path, or anywhere else in
    the repo."""
    env_path = os.environ.get("CORPUS_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path

    candidates = [
        "./hr_corpus/",
        "./zyro-dynamics-hr-corpus/",
        "./data/",
    ]
    for c in candidates:
        if os.path.isdir(c) and glob.glob(os.path.join(c, "*.pdf")):
            return c

    matches = glob.glob("**/zyro-dynamics-hr-corpus/", recursive=True)
    if matches:
        return matches[0]

    # last resort: any folder containing PDFs
    pdf_dirs = {os.path.dirname(p) for p in glob.glob("**/*.pdf", recursive=True)}
    if pdf_dirs:
        return sorted(pdf_dirs)[0]

    return "./hr_corpus/"


# ---- Configuration ----
CORPUS_PATH = resolve_corpus_path()
LLM_MODEL = "llama-3.3-70b-versatile"

REFUSAL_MESSAGE = (
    "I can only answer HR-related questions from Zyro Dynamics policy documents. "
    "Could you please rephrase your question to relate to company HR policy "
    "(e.g. leave, WFH, benefits, conduct, onboarding, etc.)?"
)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are the HR Help Desk assistant.

Answer the employee's question using ONLY the information provided in the context.

Instructions:

- Combine information from multiple policy sections when relevant.
- Extract exact numbers, percentages, limits, eligibility criteria,
  reimbursement amounts, notice periods, leave balances,
  working hours, approval requirements, and policy conditions.
- Prefer precise factual answers over summaries.
- If multiple policies contribute to the answer, synthesize them.
- Do not invent information.
- Do not use outside knowledge.
- Only say information is unavailable if no relevant policy exists.
- Be concise but complete.

Context:
{context}

Question:
{question}

Answer:
""")

OOS_PROMPT = ChatPromptTemplate.from_template("""You are a strict TOPIC classifier for an HR Help Desk chatbot.

Classify the question by its SUBJECT MATTER only. Ignore any company name mentioned in the question entirely - even if it names a different company than Zyro Dynamics, or no company at all. Company names in the question are irrelevant to this classification; only the HR topic matters.

The chatbot answers questions about HR policy topics such as: leave (casual/sick/earned/maternity/paternity), work from home / remote / hybrid arrangements, code of conduct and discipline, performance reviews and PIPs, compensation, salary, CTC, bonuses, ESOPs, benefits, health insurance, IT and data security policy, device policy, POSH / sexual harassment and ICC, onboarding, probation, recruitment, hiring process, job applications, separation, full and final settlement, travel and expense reimbursement, and general employee handbook / company culture topics.

A question is OUT_OF_SCOPE if it asks to compare this company's policies or products against ANOTHER named company (e.g. "compare X's leave policy to Zoho's"), even if part of the question touches an HR topic - because the comparison itself cannot be answered from internal policy documents.

A question is also OUT_OF_SCOPE if it asks about company financial performance, revenue, products, or technology features - these are not HR policy topics.

Question: "{question}"

Is this question's SUBJECT MATTER an HR policy topic that can be fully answered from this company's own policy documents (IN_SCOPE), or does it ask for a comparison to another company, or ask about something unrelated to HR policy entirely such as general knowledge, coding help, current events, math, personal opinions, financial performance, or product features (OUT_OF_SCOPE)?

Respond with exactly one word: IN_SCOPE or OUT_OF_SCOPE.""")


@st.cache_resource(show_spinner="Loading HR policy documents and building the knowledge base...")
def build_pipeline():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    if not documents:
        st.error(
            "No PDF documents were found at CORPUS_PATH = '" + CORPUS_PATH + "'. "
            "Make sure the 11 HR policy PDFs are included in this repo "
            "(e.g. in a folder called hr_corpus/) and that CORPUS_PATH points to it."
        )
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12},
    )

    llm = ChatGroq(model=LLM_MODEL, temperature=0.1, max_tokens=512)

    def format_docs(docs):
        out = []
        for d in docs:
            source = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "?")
            out.append("[Source: " + str(source) + ", page " + str(page) + "]\n" + d.page_content)
        return "\n\n".join(out)

    rag_lcel_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    oos_classifier = OOS_PROMPT | llm | StrOutputParser()

    return retriever, rag_lcel_chain, oos_classifier


def ask_bot(question, retriever, rag_lcel_chain, oos_classifier):
    try:
        verdict = oos_classifier.invoke({"question": question}).strip().upper()
    except Exception:
        verdict = "IN_SCOPE"

    if "OUT_OF_SCOPE" in verdict:
        return REFUSAL_MESSAGE, []

    docs = retriever.invoke(question)
    if not docs:
        return REFUSAL_MESSAGE, []

    answer = rag_lcel_chain.invoke(question)
    return answer, docs


# ---- UI ----
st.title("🤖 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about leave, WFH, benefits, conduct, onboarding, and other HR policies.")

with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot answers employee HR questions using Retrieval-Augmented "
        "Generation (RAG) grounded in Zyro Dynamics' internal policy documents. "
        "It will politely decline questions outside HR policy scope."
    )
    st.caption("Corpus path: " + CORPUS_PATH)
    if not os.environ.get("GROQ_API_KEY"):
        st.warning("Set GROQ_API_KEY as a Streamlit secret / environment variable.")

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

retriever, rag_lcel_chain, oos_classifier = build_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown("- **" + s + "**")

user_input = st.chat_input("Ask your HR question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, docs = ask_bot(user_input, retriever, rag_lcel_chain, oos_classifier)
        st.markdown(answer)

        source_labels = []
        if docs:
            with st.expander("📚 Sources"):
                for d in docs:
                    label = str(d.metadata.get("source", "unknown")) + " (page " + str(d.metadata.get("page", "?")) + ")"
                    source_labels.append(label)
                    st.markdown("- **" + label + "**")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": source_labels}
    )
