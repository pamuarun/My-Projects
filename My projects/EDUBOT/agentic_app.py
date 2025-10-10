
# -*- coding: utf-8 -*-
"""
EDUBOT - Full Streamlit App with FAISS, LLM, Multi-File Upload, Styled UI, Memory, Academic Check
"""
import os, re, time
import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredPowerPointLoader, UnstructuredExcelLoader
import evaluate
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
import sympy as sp
import math
import matplotlib.pyplot as plt
import numpy as np
import google.generativeai as genai  # For Image Agent
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter



# =========================
# CONFIG
# =========================
st.set_page_config(page_title="EDUBOT", page_icon="🤖", layout="wide")
DB_FAISS_PATH = r"D:\360\Project - 4\EDUBOT\vectorstore"
DATA_PATH = r"D:\360\Project - 4\EDUBOT\Data"
API_KEY = " "

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# =========================
# Load FAISS DB
# =========================
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(DB_FAISS_PATH):
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS(embeddings.embed_query, embeddings.embed_documents, [])
    return db, db.as_retriever(search_kwargs={"k": 10}), embeddings

db, retriever, embeddings = load_retriever()



def process_and_store_text(text_content, filename, db, embeddings):
    """
    Split text into chunks and store in FAISS DB
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,    # adjust between 500–1000
        chunk_overlap=100  # keeps continuity between chunks
    )
    chunks = text_splitter.split_text(text_content)
    
    if chunks:
        db.add_texts(chunks)
        db.save_local(DB_FAISS_PATH)
        st.success(f"✅ File '{filename}' added ({len(chunks)} chunks stored in vector DB)")
    else:
        st.info(f"⚠ File '{filename}' had no readable content.")


# ============================ #
# Step 9: Utilities
# ============================ #
def format_numeric_value(val, decimals=4):
    try:
        if val == sp.zoo:
            return "undefined (infinite)"
        if val == sp.oo:
            return "infinity"
        if val == -sp.oo:
            return "-infinity"
        f = float(val)
        if math.isinf(f):
            return "infinity" if f > 0 else "-infinity"
        if math.isnan(f):
            return "undefined"
        if abs(f - int(f)) < 1e-12:
            return str(int(round(f)))
        s = f"{round(f, decimals):.{decimals}f}"
        s = s.rstrip('0').rstrip('.')
        return s
    except Exception:
        return str(val)

def safe_sympify(expr_text):
    try:
        expr_text = expr_text.replace("^", "**")
        return sp.sympify(expr_text)
    except Exception:
        try:
            val = float(expr_text)
            return sp.N(val)
        except Exception:
            raise

# ============================ #
# Step 10: Wolfram Agent (FULL pretty Unicode output)
# ============================ #
def wolfram_agent(query):
    x = sp.symbols('x')
    ql = query.strip()

    SUPERSCRIPTS = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'}
    import re
    def pretty_power(expr):
        s = str(expr)
        def repl(m):
            base, exp = m.group(1), m.group(2)
            exp_sup = ''.join(SUPERSCRIPTS.get(c, c) for c in exp)
            return f"{base}{exp_sup}"
        s = re.sub(r'(\w+)\*\*(\d+)', repl, s)
        s = re.sub(r'sp\.sqrt\((.*?)\)', r'√(\1)', s)
        s = s.replace('sqrt','√')
        s = s.replace('**','^')
        return s

    def recursive_eval(expr):
        if isinstance(expr, sp.Add):
            vals, steps = [], []
            for term in sp.Add.make_args(expr):
                val, substeps = recursive_eval(term)
                vals.append(val)
                steps.extend(substeps)
            total = sum(vals)
            steps.append(f"{' + '.join([format_numeric_value(v) for v in vals])} = {format_numeric_value(total)}")
            return total, steps

        elif isinstance(expr, sp.Mul):
            vals, steps = [], []
            for factor in sp.Mul.make_args(expr):
                val, substeps = recursive_eval(factor)
                vals.append(val)
                steps.extend(substeps)
            total = 1
            for v in vals:
                total *= v
            steps.append(f"{' × '.join([format_numeric_value(v) for v in vals])} = {format_numeric_value(total)}")
            return total, steps

        elif isinstance(expr, sp.Pow):
            base, exp = expr.as_base_exp()
            val_base, steps_base = recursive_eval(base)
            val_exp, steps_exp = recursive_eval(exp)
            val = val_base ** val_exp
            steps = steps_base + steps_exp
            steps.append(f"{format_numeric_value(val_base)}^{format_numeric_value(val_exp)} = {format_numeric_value(val)}")
            return val, steps

        else:
            val = float(expr)
            return val, [f"{pretty_power(expr)} = {format_numeric_value(val)}"]

    try:
        lowered = ql.lower()

        # Solve equations
        if lowered.startswith("solve") or " solve " in lowered:
            eq_match = re.search(r"solve\s+(.*)\s*=\s*(.*)", ql, flags=re.I)
            if eq_match:
                lhs_text = eq_match.group(1).strip()
                rhs_text = eq_match.group(2).strip()
                expr = safe_sympify(lhs_text) - safe_sympify(rhs_text)
                poly = sp.expand(expr)
                sol = sp.solve(expr, x)
                if poly.is_polynomial(x) and sp.degree(poly, x) == 2:
                    a, b, c = sp.Poly(poly, x).all_coeffs()
                    disc = sp.simplify(b**2 - 4*a*c)
                    sqrt_disc = sp.sqrt(disc)
                    root1 = sp.simplify((-b + sqrt_disc) / (2*a))
                    root2 = sp.simplify((-b - sqrt_disc) / (2*a))
                    steps = [
                        "[Wolfram Agent] Quadratic Solution Steps:",
                        f"Step 1: Original Equation: {lhs_text} = {rhs_text}",
                        f"Step 2: Bring all terms to LHS: {pretty_power(poly)} = 0",
                        f"Step 3: Identify coefficients: a={format_numeric_value(a)}, b={format_numeric_value(b)}, c={format_numeric_value(c)}",
                        f"Step 4: Compute discriminant: Δ = b² - 4ac = {pretty_power(disc)}",
                        f"Step 5: √Δ = {pretty_power(sqrt_disc)}",
                        f"Step 6: Apply quadratic formula: x = (-b ± √Δ) / 2a",
                        f"Step 7: Calculate roots: x₁ = {pretty_power(root1)}, x₂ = {pretty_power(root2)}",
                        f"✅ Final Answer: {pretty_power(sol)}"
                    ]
                    return "\n".join(steps)
                else:
                    return f"[Wolfram Agent] Solution: {pretty_power(sol)}"

        # Derivative
        if re.search(r"\b(derivative|differentiate)\b", lowered):
            m = re.search(r"(?:derivative of|differentiate|derivative)\s*(.*)", ql, flags=re.I)
            expr_text = m.group(1).strip() if m else ql.split(":",1)[-1].strip()
            expr = safe_sympify(expr_text)
            derivative = sp.diff(expr, x)
            steps = ["[Wolfram Agent] Derivative Steps:", f"Function: f(x) = {pretty_power(expr)}"]
            for term in sp.Add.make_args(sp.expand(expr)):
                d = sp.diff(term, x)
                steps.append(f" - d/dx({pretty_power(term)}) = {pretty_power(d)}")
            steps.append(f"✅ Final Derivative: f'(x) = {pretty_power(derivative)}")
            return "\n".join(steps)

        # Integration
        if re.search(r"\b(integrate|integral|integration)\b", lowered):
            m = re.search(r"(?:integrate|integral of|integration of)\s*(.*)", ql, flags=re.I)
            expr_text = m.group(1).strip() if m else None
            expr = safe_sympify(expr_text)
            integral = sp.integrate(expr, x)
            steps = ["[Wolfram Agent] Integration Steps:", f"Function: f(x) = {pretty_power(expr)}"]
            for term in sp.Add.make_args(sp.expand(expr)):
                integ = sp.integrate(term, x)
                steps.append(f" - ∫({pretty_power(term)}) dx = {pretty_power(integ)}")
            steps.append(f"✅ Final Integral: ∫f(x) dx = {pretty_power(integral)} + C")
            return "\n".join(steps)

        # Compute/Evaluate
        calc_match = re.match(r'^(compute|calculate|evaluate|find)\s+(.*)', lowered)
        if calc_match:
            expr_text = calc_match.group(2).strip()
            sub_match = re.search(r'(.*)\b(?:at|for)\s*x\s*=\s*([-\d\./]+)\b', expr_text)
            if sub_match:
                expr_only = sub_match.group(1).strip()
                x_val_str = sub_match.group(2)
                x_val = float(sp.N(safe_sympify(x_val_str)))
                expr_sym = sp.sympify(expr_only, evaluate=False)
                terms = sp.Add.make_args(sp.expand(expr_sym))
                term_vals = []
                steps = ["[Wolfram Agent] Step-by-Step Computation (with substitution):"]
                steps.append(f"Step 1: Original Expression: {pretty_power(expr_only)}")
                for t in terms:
                    substituted_term = t.subs(x, x_val)
                    evaluated_term = sp.N(substituted_term)
                    steps.append(f" - Substitute x={format_numeric_value(x_val)} in {pretty_power(t)} → {format_numeric_value(evaluated_term)}")
                    term_vals.append(evaluated_term)
                total = sum(term_vals)
                steps.append(f"Step 2: Add all evaluated terms: {' + '.join([format_numeric_value(v) for v in term_vals])} = {format_numeric_value(total)}")
                steps.append(f"✅ Final Answer = {format_numeric_value(total)}")
                return "\n".join(steps)
            else:
                expr_sym = sp.sympify(expr_text, evaluate=False)
                total, detailed_steps = recursive_eval(expr_sym)
                steps = ["[Wolfram Agent] Step-by-Step Computation:"] + detailed_steps
                steps.append(f"✅ Final Answer = {format_numeric_value(total)}")
                return "\n".join(steps)

        # Plot
        if re.search(r"\bplot\b", lowered):
            m = re.search(r"plot\s+(.*)", ql, flags=re.I)
            expr_text = m.group(1).strip() if m else "x**2 - 4*x + 3"
            expr = safe_sympify(expr_text)

            # Step-by-step evaluation
            xs_steps = np.linspace(-10, 10, 10)  # fewer points for explanation
            steps = [f"[Wolfram Agent] Step-by-Step Plot Evaluation for {expr_text}:"]
            ys_steps = []
            for xv in xs_steps:
                yv = float(sp.N(expr.subs(x, xv)))
                ys_steps.append(yv)
                steps.append(f" - x = {format_numeric_value(xv)} → f(x) = {format_numeric_value(yv)}")

            # Full plot
            xs_plot = np.linspace(-10, 10, 400)
            ys_plot = [float(sp.N(expr.subs(x, xv))) for xv in xs_plot]
            plt.figure(figsize=(6,4))
            plt.plot(xs_plot, ys_plot, label=expr_text)
            plt.title(f"[Wolfram Agent] Plot of {expr_text}")
            plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.legend()

            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)

            steps.append(f"✅ Plot generated for {expr_text}")
            return {"text": "\n".join(steps), "plot": buf}



        return None

    except Exception as e:
        return f"[Wolfram Agent] Error: {e}"


# ============================ #
# Step 11: Quiz / MCQ Agent
# ============================ #
def quiz_agent(query, context_text):
    try:
        if "mcq" not in query.lower() and "quiz" not in query.lower():
            return None
        match = re.search(r"(\d+)", query)
        num_mcqs = int(match.group(1)) if match else 3
        prompt = f"""
You are an academic MCQ generator for K-12 students.
Generate exactly {num_mcqs} MCQs on the topic below.
Topic/context:
{context_text if context_text else query}
"""
        llm_response = llm(prompt)
        if isinstance(llm_response, dict):
            answer_text = llm_response.get("output_text") or llm_response.get("text") or str(llm_response)
        else:
            answer_text = getattr(llm_response, "output_text", None) or getattr(llm_response, "text", None) or str(llm_response)
        return f"[Quiz Agent] Generated {num_mcqs} MCQs:\n{answer_text.strip()}"
    except Exception as e:
        return f"[Quiz Agent] Error: {e}"


# ============================ #
# Step 12: Image Agent (Gemini + Hugging Face Backup)
# ============================ #
from huggingface_hub import InferenceClient
import base64

# Hugging Face Backup Client
HF_TOKEN = " "
hf_client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=HF_TOKEN)

genai.configure(api_key=API_KEY)

def image_agent(prompt):
    """
    Generate an educational image using:
    1. Google Gemini (primary)
    2. Hugging Face Stable Diffusion XL (backup if Gemini quota fails)
    Returns: dict with 'text' and 'image' (PIL Image)
    """
    full_prompt = f"High-quality educational diagram for K-12 students: {prompt}, clear, neat"

    # ---------- Try Gemini First ----------
    try:
        print("🟢 Trying Gemini Image API...")
        model = genai.GenerativeModel("gemini-2.5-flash-preview-image")
        response = model.generate_content(full_prompt)

        for idx, candidate in enumerate(response.candidates):
            for part in candidate.content.parts:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    img_data = part.inline_data.data
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(BytesIO(img_bytes))
                    return {
                        "text": f"[Image Agent] ✅ Gemini image generated",
                        "image": img
                    }

    except Exception as e:
        print(f"⚠️ Gemini failed: {e}")

    # ---------- Hugging Face Backup ----------
    try:
        print("🟡 Falling back to Hugging Face Stable Diffusion...")
        image = hf_client.text_to_image(prompt=full_prompt)
        return {
            "text": f"[Image Agent] ✅ Hugging Face image generated",
            "image": image
        }
    except Exception as e:
        return {
            "text": f"[Image Agent] ❌ Both Gemini & Hugging Face failed. Error: {e}",
            "image": None
        }




# =========================
# LLM & Prompt Template
# =========================
llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=API_KEY,
    temperature=0.2,
    max_output_tokens=1200
)

FULL_PROMPT_TEMPLATE = """
You are EDUBOT, an AI tutor for K-12 students. 
Act like a flexible teacher who adapts explanations to the student’s intent.

---

### Memory & Context Rules:
- Always use **chat history** to interpret vague follow-ups (e.g., "it", "this", "go with that").
- Continue the flow instead of repeating the same explanation.
- If FAISS context is weak, **fallback to general academic knowledge**.
- If the student gives acknowledgments like "okay", "yes", "continue", interpret them as **follow-up requests**.

---

### Question-Type Rules (strict):
- If the question starts with **Who** → answer only *who (person, group, entity)* with background, role, contributions, legacy.
- If the question starts with **What** → answer only *what (definition, fact, meaning)* with scope, uses, and applications.
- If the question starts with **When** → answer only *time-related details*.
- If the question starts with **Why** → answer only *reasons/importance*.
- If the question starts with **How** → answer only *steps, process, or explanation*.
- Do not mix categories unless the student explicitly asks.

---

### Off-Topic Rules (very strict):
- 🚫 Do **NOT** answer questions about:
  - Movies, actors, or celebrities
  - Jokes, memes, or humor requests
  - Politics, political leaders, or elections
  - Personal/private questions unrelated to academics
- Instead, politely respond: *"This question is not related to your study material. Please ask me something academic."*

---

### Depth Control:
- Always expand answers into **at least 4–5 lines**.
- Provide examples, applications, and relevant context.
- Avoid irrelevant details.

---

Chat History:
{chat_history}

Context from study material:
{context}

Student Question:
{question}

Answer:
"""

prompt = PromptTemplate(template=FULL_PROMPT_TEMPLATE, input_variables=["chat_history","context","question"])
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=False,
    output_key="answer"
)

# =========================
# Evaluation
# =========================
bleu_metric = evaluate.load("bleu")
rouge_metric = Rouge()

def semantic_similarity_score(reference, generated, embed_model=embeddings):
    if not reference.strip() or not generated.strip():
        return None
    ref_vec = embed_model.embed_query(reference)
    gen_vec = embed_model.embed_query(generated)
    score = cosine_similarity([ref_vec], [gen_vec])[0][0]
    return round(score,4)

def evaluate_response(reference, generated):
    scores = {"BLEU": None, "ROUGE": None, "SemanticSim": None}
    if reference and reference.strip():
        scores["BLEU"] = bleu_metric.compute(predictions=[generated], references=[[reference]])["bleu"]
        rouge_scores = rouge_metric.get_scores(generated, reference)[0]
        scores["ROUGE"] = rouge_scores
        scores["SemanticSim"] = semantic_similarity_score(reference, generated)
    return scores

# =========================
# Academic Question Check
# =========================
def is_academic_question(question):
    followups = r"\b(ok|okay|yes|continue|go with this|that one|steps in it|the 3rd one)\b"
    if re.search(followups, question.lower()):
        return True
    non_academic_patterns = [
        r"\b(joke|funny|politics|movie|celebrity|personal)\b",
        r"\b(who|where|when) is .* president\b"
    ]
    for pat in non_academic_patterns:
        if re.search(pat, question.lower()):
            return False
    return True


# =========================
# Styling
# =========================
st.markdown("""
<style>
.stApp, .stApp > .css-18e3th9, .stApp > .block-container,
.stApp > .main > div, .stApp > div[role="main"] > div:last-child {background-color: #24142b !important; color: #FFFFFF !important;}
header, .css-1v3fvcr, .css-18e3th9 { background-color: #24142b !important; color: #FFFFFF !important;}
[data-testid="stSidebar"] { background-color: #2a023f !important; color: #FFFFFF !important; }
.sidebar-section { background-color: #a707f7 !important; color: #FFFFFF !important; padding: 8px 12px !important;
border-radius: 8px !important; margin-bottom: 12px !important; font-weight: bold !important;
transition: transform 0.3s, box-shadow 0.3s; }
.sidebar-section:hover { transform: translateY(-3px); box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
.stButton>button, div.stFileUploader>div>button { background-color: #FF0000 !important; color: #FFFFFF !important;
border: none !important; width: 100% !important; font-weight: bold !important;
transition: transform 0.3s, box-shadow 0.3s; }
.stButton>button:hover, div.stFileUploader>div>button:hover { transform: translateY(-3px);
box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
.header-title { color: #FFFFFF !important; font-size: 60px !important; font-weight: bold !important; text-align: center !important; }
.sub-title { color: #FFFFFF !important; text-align: center !important; font-size: 24px !important; margin-bottom: 20px !important; }
.chat-bubble { display: inline-block; padding: 12px 16px; border-radius: 16px; margin: 6px 0;
max-width: 80%; word-wrap: break-word; font-size: 16px; }
.user-bubble { background-color: #a707f7 !important; color: white !important; }
.assistant-bubble { background-color: #F7D6E0 !important; color: #000 !important; }
.user-icon, .bot-icon { width:32px; height:32px; background-size: contain; display:inline-block; margin-right:8px; }
.user-icon { background-image: url('https://img.icons8.com/color/48/user.png'); }  
.bot-icon { background-image: url('https://img.icons8.com/color/48/bot.png'); }
.flex-container { display:flex; align-items:flex-start; margin-bottom:8px; }
.flex-end { justify-content:flex-end; }
.flex-start { justify-content:flex-start; }
.st-chat-input, .st-chat-input textarea, .st-chat-input div[role="textbox"] { background-color: #24142b !important;
color: #FFFFFF !important; border: 1px solid #a707f7 !important; border-radius: 12px !important; padding: 8px !important; }
.st-chat-input button[type="submit"] { background-color: #a707f7 !important; color: #FFFFFF !important;
border-radius: 12px !important; font-weight: bold !important; transition: transform 0.2s, box-shadow 0.2s; }
.st-chat-input button[type="submit"]:hover { background-color: #FF66FF !important; transform: translateY(-2px);
box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
.st-chat-input::after { content: ""; display: block; height: 20px; background-color: #24142b !important; }
.message-history-item { background-color: rgba(167,7,247,0.2) !important; padding: 6px 8px !important;
border-radius: 6px !important; margin-bottom: 4px !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("<div class='header-title'>🤖 EDUBOT - Your Study Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Learn Smarter, Not Harder!</div>", unsafe_allow_html=True)

# =========================
# Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# For storing generated images
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []
    
    
# =========================
# Sidebar (Controls Only) - Fixed with Proper Message Placement
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-section">📁 Upload Files</div>', unsafe_allow_html=True)
    
    # --- File uploader with dynamic key ---
    uploaded_files = st.file_uploader(
        "", 
        type=["txt","pdf","ppt","pptx","doc","docx","xls","xlsx"], 
        label_visibility="collapsed", 
        accept_multiple_files=True,
        key=st.session_state.get("files_uploader_key", "files_uploader")
    )

    files_msg_container = st.container()

    st.markdown('<div class="sidebar-section">🖼️ Upload Images</div>', unsafe_allow_html=True)
    
    # --- Image uploader with dynamic key ---
    uploaded_images = st.file_uploader(
        "", 
        type=["jpg","jpeg","png"], 
        label_visibility="collapsed", 
        accept_multiple_files=True,
        key=st.session_state.get("images_uploader_key", "images_uploader")
    )

    images_msg_container = st.container()

    # --- Initialize session trackers ---
    if "summarized_files" not in st.session_state:
        st.session_state.summarized_files = set()
    if "summarized_images" not in st.session_state:
        st.session_state.summarized_images = set()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files_state" not in st.session_state:
        st.session_state.uploaded_files_state = []
    if "uploaded_images_state" not in st.session_state:
        st.session_state.uploaded_images_state = []

    # --- Track uploads via session state ---
    if uploaded_files:
        st.session_state.uploaded_files_state = uploaded_files
    if uploaded_images:
        st.session_state.uploaded_images_state = uploaded_images

    # --- Display file upload messages ---
    if st.session_state.uploaded_files_state:
        with files_msg_container:
            for uploaded_file in st.session_state.uploaded_files_state:
                if uploaded_file.name in st.session_state.summarized_files:
                    st.success(f"✅ File '{uploaded_file.name}' already summarized.")
                else:
                    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

    # --- Display image upload messages ---
    if st.session_state.uploaded_images_state:
        with images_msg_container:
            for uploaded_image in st.session_state.uploaded_images_state:
                if uploaded_image.name in st.session_state.summarized_images:
                    st.success(f"✅ Image '{uploaded_image.name}' already processed.")
                else:
                    st.success(f"✅ Image '{uploaded_image.name}' uploaded successfully!")

    # --- Chats section ---
    st.markdown('<div class="sidebar-section">💬 Chats</div>', unsafe_allow_html=True)
    if st.button("➕ New Chat"):
        # Clear session state
        st.session_state.messages = []
        st.session_state.summarized_files = set()
        st.session_state.summarized_images = set()
        st.session_state.uploaded_files_state = []
        st.session_state.uploaded_images_state = []

        # Reset uploader widgets by changing keys
        st.session_state["files_uploader_key"] = f"files_uploader_{time.time()}"
        st.session_state["images_uploader_key"] = f"images_uploader_{time.time()}"

        files_msg_container.empty()
        images_msg_container.empty()
        st.success("Started a new chat!")

    # --- Message History section ---
    st.markdown('<div class="sidebar-section">🕘 Message History</div>', unsafe_allow_html=True)
    for idx, msg in enumerate(st.session_state.messages):
        st.markdown(
            f"<div class='message-history-item'>{idx+1}. {msg['role'].capitalize()}: {msg['content']}</div>", 
            unsafe_allow_html=True
        )

    # --- Logout button ---
    if st.button("🔒 Logout"):
        # Clear session state
        st.session_state.messages = []
        st.session_state.summarized_files = set()
        st.session_state.summarized_images = set()
        st.session_state.uploaded_files_state = []
        st.session_state.uploaded_images_state = []

        # Reset uploader widgets by changing keys
        st.session_state["files_uploader_key"] = f"files_uploader_{time.time()}"
        st.session_state["images_uploader_key"] = f"images_uploader_{time.time()}"

        files_msg_container.empty()
        images_msg_container.empty()
        st.success("You have logged out successfully! 🙂")

# =========================
# Main Area: Use session state uploads
# =========================
uploaded_files = st.session_state.uploaded_files_state
uploaded_images = st.session_state.uploaded_images_state

# =========================
# Main Area: File & Image Processing with Animated Summary
# =========================

# === FILES ===
if uploaded_files:
    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        ext = os.path.splitext(uploaded_file.name)[1].lower()

        if uploaded_file.name in st.session_state.summarized_files:
            continue

        text_content, summary = "", ""

        try:
            # Extract Text
            if ext == ".txt":
                text_content = open(temp_file_path, "r", encoding="utf-8").read()
            elif ext == ".pdf":
                text_content = "".join([p.extract_text() or "" for p in PdfReader(temp_file_path).pages])
            elif ext in [".doc", ".docx"]:
                text_content = "\n".join([d.page_content for d in Docx2txtLoader(temp_file_path).load()])
            elif ext in [".ppt", ".pptx"]:
                text_content = "\n".join([d.page_content for d in UnstructuredPowerPointLoader(temp_file_path).load()])
            elif ext in [".xls", ".xlsx"]:
                text_content = "\n".join([d.page_content for d in UnstructuredExcelLoader(temp_file_path).load()])

            if text_content.strip():
                # Add to FAISS DB
                db.add_texts([text_content])
                db.save_local(DB_FAISS_PATH)

                # Generate summary
                summary_prompt = f"Summarize the following text for study notes in 4–5 lines:\n\n{text_content[:1500]}"
                try:
                    summary = llm.invoke(summary_prompt).strip()
                except Exception:
                    summary = "⚠️ Could not generate summary."

                st.session_state.summarized_files.add(uploaded_file.name)

                # Animated summary in main chat
                summary_msg = f"Here’s a quick summary of *{uploaded_file.name}*:\n\n{summary}"
                placeholder = st.empty()
                typed_text = ""
                for char in summary_msg:
                    typed_text += char
                    placeholder.markdown(
                        f"""<div class="flex-container flex-start">
                        <div class='bot-icon' style="background-image:url('https://img.icons8.com/color/48/opened-folder.png');"></div>
                        <div class='chat-bubble assistant-bubble'>{typed_text}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                # Store with icon + agent info
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": summary_msg,
                    "agent": "File Agent",
                    "icon": "https://img.icons8.com/color/48/opened-folder.png"
                })
                memory.chat_memory.add_ai_message(summary_msg)

            else:
                st.warning(f"⚠️ File '{uploaded_file.name}' has no extractable text.")

        except Exception as e:
            st.error(f"Failed to process {uploaded_file.name}: {e}")

# === IMAGES ===
if uploaded_images:
    import easyocr
    from transformers import BlipProcessor, BlipForConditionalGeneration

    # Initialize EasyOCR & BLIP
    reader = easyocr.Reader(['en'])
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    for uploaded_image in uploaded_images:
        if uploaded_image.name in st.session_state.summarized_images:
            continue

        img = Image.open(uploaded_image).convert("RGB")
        img_np = np.array(img)

        # ---------- OCR Stage ----------
        try:
            ocr_result = reader.readtext(img_np)
            text_content = " ".join([res[1] for res in ocr_result]).strip()
        except Exception:
            text_content = ""

        # ---------- BLIP Caption ----------
        try:
            inputs = processor(images=img, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        except Exception:
            caption = "⚠️ Could not generate image caption."

        # ---------- Summarize Text if OCR detected ----------
        if text_content:
            summary_prompt = f"Summarize the following text for study notes in 4–5 lines:\n\n{text_content[:1500]}"
            try:
                text_summary = llm.invoke(summary_prompt).strip()
            except Exception:
                text_summary = "⚠️ Could not generate text summary."
            final_msg = f"🖼️ *Image Caption:* {caption}\n\n📝 *Text Summary:* {text_summary}"
        else:
            final_msg = f"🖼️ *Image Caption / Description:* {caption}\n\n📝 No text detected via OCR."

        # ---------- Animated typing effect ----------
        placeholder = st.empty()
        typed_text = ""
        for char in final_msg:
            typed_text += char
            placeholder.markdown(
                f"""<div class="flex-container flex-start">
                <div class='bot-icon' style="background-image:url('https://img.icons8.com/color/48/image.png');"></div>
                <div class='chat-bubble assistant-bubble'>{typed_text}</div>
                </div>""",
                unsafe_allow_html=True
            )

        # ---------- Update Session State ----------
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_msg,
            "agent": "Image Caption & Text Agent",
            "icon": "https://img.icons8.com/color/48/image.png"
        })
        memory.chat_memory.add_ai_message(final_msg)
        st.session_state.summarized_images.add(uploaded_image.name)

        # ---------- Success Message under uploader ----------
        images_msg_container.success(f"✅ Image '{uploaded_image.name}' processed successfully!")


#------DISPLAY---------
def display_message(msg):
    if msg["role"] == "user":
        st.markdown(f"""<div class="flex-container flex-end">
            <div class='chat-bubble user-bubble'>{msg['content'].replace(chr(10), '<br>')}</div>
            <div class='user-icon'></div>
        </div>""", unsafe_allow_html=True)
    else:
        icon_url = msg.get("icon", "https://img.icons8.com/color/48/bot.png")
        st.markdown(f"""<div class="flex-container flex-start">
            <div class='bot-icon' style="background-image:url('{icon_url}')"></div>
            <div class='chat-bubble assistant-bubble'>{msg['content'].replace(chr(10), '<br>')}</div>
        </div>""", unsafe_allow_html=True)

        # Display plot if available
        if msg.get("plot"):
            st.image(msg["plot"], caption="Plot", use_container_width=True)
        if msg.get("image"):
            st.image(msg["image"], caption="Generated Image", use_container_width=True)

for msg in st.session_state.messages:
    display_message(msg)
    

# =========================
# Chat Input with Dynamic Agents, Icons, and Inline Images/Plots
# =========================
user_input = st.chat_input("Ask me anything about your subjects...")

if user_input:
    # Add user message to session
    st.session_state.messages.append({"role": "user", "content": user_input})
    display_message({"role": "user", "content": user_input})

    # Farewell handling
    if user_input.lower() in ["bye", "quit", "exit"]:
        farewell_msg = "Thank you! Have a great day! 🙂 If you have more questions later, just ask!"
        st.session_state.messages.append({
            "role": "assistant",
            "content": farewell_msg,
            "agent": "EDUBOT (LLM)",
            "icon": "https://img.icons8.com/color/48/bot.png"
        })
        display_message({
            "role": "assistant",
            "content": farewell_msg,
            "icon": "https://img.icons8.com/color/48/bot.png"
        })
    else:
        # Academic check
        if not is_academic_question(user_input):
            answer_lines = ["This question is not related to the study material."]
            agent_name = "EDUBOT (LLM)"
            plot_buf = None
            img_buf = None
        else:
            # Sync LangChain memory with session messages
            memory.chat_memory.messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    memory.chat_memory.add_user_message(msg["content"])
                else:
                    memory.chat_memory.add_ai_message(msg["content"])

            answer_lines = []
            agent_name = "EDUBOT (LLM)"
            plot_buf = None
            img_buf = None

            # 1. Wolfram Agent
            wolfram_ans = wolfram_agent(user_input)
            if wolfram_ans:
                if isinstance(wolfram_ans, dict):
                    answer_lines = [wolfram_ans.get("text", "")]
                    plot_buf = wolfram_ans.get("plot", None)
                else:
                    answer_lines = wolfram_ans.split("\n")
                    plot_buf = None
                agent_name = "Wolfram Agent"

            # 2. Quiz Agent
            elif "quiz" in user_input.lower() or "mcq" in user_input.lower():
                quiz_ans = quiz_agent(user_input, "")
                if quiz_ans:
                    if isinstance(quiz_ans, dict):
                        answer_lines = [quiz_ans.get("text", "")]
                    else:
                        answer_lines = quiz_ans.split("\n")
                    agent_name = "Quiz Agent"

            # 3. Image Generation Agent
            elif any(word in user_input.lower() for word in ["diagram", "draw", "illustrate", "image", "generate"]):
                img_ans = image_agent(user_input)
                img_buf = None
                if img_ans:
                    if isinstance(img_ans, dict):
                        answer_lines = [img_ans.get("text", "")]
                        img_data = img_ans.get("image", None)
                        if img_data:
                            try:
                                # Convert to PIL image if necessary
                                if isinstance(img_data, str) and os.path.exists(img_data):
                                    img_buf = Image.open(img_data)
                                elif isinstance(img_data, bytes):
                                    
                                    img_buf = Image.open(BytesIO(img_data))
                                else:
                                    img_buf = img_data  # already PIL Image
                            except Exception as e:
                                st.warning(f"⚠️ Could not load generated image: {e}")
                    else:
                        answer_lines = [img_ans]
                    agent_name = "Image Agent"

            # 4. Fallback LLM
            if not answer_lines:
                result = qa_chain({"question": user_input})
                answer_lines = [result["answer"]]
                agent_name = "EDUBOT (LLM)"

        # =========================
        # Agent icons mapping
        # =========================
        agent_icons = {
            'Wolfram Agent': 'https://img.icons8.com/color/48/calculator.png',
            'Quiz Agent': 'https://img.icons8.com/color/48/light.png',
            'Image Agent': 'https://img.icons8.com/color/48/picture.png',
            'Image Caption & Text Agent': 'https://img.icons8.com/color/48/image.png',
            'EDUBOT (LLM)': 'https://img.icons8.com/color/48/bot.png',
            'File Agent': 'https://img.icons8.com/color/48/opened-folder.png'
        }
        icon_url = agent_icons.get(agent_name, agent_icons['EDUBOT (LLM)'])

        # Combine all text lines into one block
        full_text = "\n".join(answer_lines)
        placeholder = st.empty()
        typed_text = ""

        # Animated typing effect
        for char in full_text:
            typed_text += char
            placeholder.markdown(
                f"""<div class="flex-container flex-start">
                    <div class='bot-icon' style="background-image:url('{icon_url}')"></div>
                    <div class='chat-bubble assistant-bubble'>{typed_text.replace(chr(10), '<br>')}</div>
                </div>""",
                unsafe_allow_html=True
            )

        # Save message to session
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_text,
            "plot": plot_buf,
            "image": img_buf,
            "agent": agent_name,
            "icon": icon_url
        })
        memory.chat_memory.add_ai_message(full_text)

        # =========================
        # Display plot or generated image if available
        # =========================
        if plot_buf:
            st.image(plot_buf, caption="Wolfram Agent Plot", use_container_width=True)
        if img_buf:
            st.image(img_buf, caption="Generated Image", use_container_width=True)
