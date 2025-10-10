
# -*- coding: utf-8 -*-
"""
FULL AGENTIC EDUBOT
Memory + Wolfram + Dynamic Quiz + CSV logging + Image Generation (Google Gemini)
Calculator logic merged into Wolfram
Interactive debug/history toggles
Updated: 2025-09-08
@author: Arun
"""

# ============================ #
# Step 0: Imports
# ============================ #
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import csv
import re
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
import evaluate
import math
from PIL import Image
from io import BytesIO
import google.generativeai as genai  # For Image Agent

# ============================ #
# Step 1: API Keys
# ============================ #
API_KEY = " "  # Google Gemini
# ============================ #
# Step 2: Load FAISS DB + Embeddings
# ============================ #
DB_FAISS_PATH = r"D:\360\Project - 4\EDUBOT\vectorstore"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 10})

# ============================ #
# Step 3: Google Gemini LLM
# ============================ #
llm = GoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=API_KEY,
    temperature=0.2,
    max_output_tokens=1500
)


# ============================ #
# Step 4: Prompt Template
# ============================ #
FULL_PROMPT_TEMPLATE = """
You are EDUBOT, an AI tutor for K-12 students. 
Act like a flexible teacher who adapts explanations to the student’s intent.

---

### Memory & Context Rules:
- Always use **chat history** to interpret vague follow-ups.
- Continue the flow instead of repeating explanations.
- If FAISS context is weak, **fallback to general academic knowledge**.
- Acknowledgments like "okay", "yes" → treat as follow-ups.

---

### Question-Type Rules:
- Who → person/group/entity
- What → definition/fact/meaning
- When → time-related
- Why → reasons/importance
- How → steps/process/explanation
- Do not mix unless explicitly asked.

---

### Depth Control:
- Expand answers into **4–5 lines minimum**.
- Provide examples, applications, relevant context.

---

Chat History:
{chat_history}

Context from study material:
{context}

Student Question:
{question}

Answer:
"""

FULL_PROMPT = PromptTemplate(
    template=FULL_PROMPT_TEMPLATE,
    input_variables=["chat_history", "context", "question"]
)

# ============================ #
# Step 5: Memory
# ============================ #
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# ============================ #
# Step 6: Conversational Retrieval Chain
# ============================ #
pdf_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": FULL_PROMPT},
    return_source_documents=False,
    output_key="answer"
)

# ============================ #
# Step 7: Evaluation Metrics
# ============================ #
bleu_metric = evaluate.load("bleu")
rouge_metric = Rouge()

def semantic_similarity_score(reference, generated, embed_model=embeddings):
    if not reference or not generated:
        return None
    try:
        ref_vec = embed_model.embed_query(reference)
        gen_vec = embed_model.embed_query(generated)
        score = cosine_similarity([ref_vec], [gen_vec])[0][0]
        return round(score, 4)
    except Exception:
        return None

def evaluate_response(reference, generated):
    scores = {"BLEU": None, "ROUGE": None, "SemanticSim": None}
    if reference and reference.strip() and generated and generated.strip():
        try:
            scores["BLEU"] = bleu_metric.compute(predictions=[generated], references=[[reference]])["bleu"]
        except Exception:
            scores["BLEU"] = None
        try:
            scores["ROUGE"] = rouge_metric.get_scores(generated, reference)[0]
        except Exception:
            scores["ROUGE"] = None
        scores["SemanticSim"] = semantic_similarity_score(reference, generated)
    return scores

# ============================ #
# Step 8: Academic Question Heuristic
# ============================ #
def is_academic_question(question):
    if not question or not question.strip():
        return False
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
    """
    Full Wolfram-style agent for EDUBOT.
    Handles:
    - Solve equations (quadratic & generic)
    - Arithmetic & Trigonometry with teacher-style step-by-step explanation
    - Derivative
    - Integration
    - Plotting
    - Substitutions (like 'at x=...')
    - Step-by-step formatting with superscripts and √ symbols
    """
    x = sp.symbols('x')
    ql = query.strip()

    # ---------------- Utility to format powers and sqrt nicely ----------------
    SUPERSCRIPTS = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'}
    import re
    def pretty_power(expr):
        """Convert ** to superscripts and sqrt() to √()"""
        s = str(expr)
        # Replace powers like x**2 → x²
        def repl(m):
            base, exp = m.group(1), m.group(2)
            exp_sup = ''.join(SUPERSCRIPTS.get(c, c) for c in exp)
            return f"{base}{exp_sup}"
        s = re.sub(r'(\w+)\*\*(\d+)', repl, s)
        # Replace sqrt() with √()
        s = re.sub(r'sp\.sqrt\((.*?)\)', r'√(\1)', s)
        s = s.replace('sqrt','√')
        # Replace ** with ^ just in case
        s = s.replace('**','^')
        return s

    # ---------------- Recursive evaluation for arithmetic ----------------
    def recursive_eval(expr):
        if isinstance(expr, sp.Add):
            vals = []
            steps = []
            for term in sp.Add.make_args(expr):
                val, substeps = recursive_eval(term)
                vals.append(val)
                steps.extend(substeps)
            total = sum(vals)
            steps.append(f"{' + '.join([format_numeric_value(v) for v in vals])} = {format_numeric_value(total)}")
            return total, steps

        elif isinstance(expr, sp.Mul):
            vals = []
            steps = []
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
            # Use superscript formatting
            steps.append(f"{format_numeric_value(val_base)}^{format_numeric_value(val_exp)} = {format_numeric_value(val)}")
            return val, steps

        else:
            val = float(expr)
            return val, [f"{pretty_power(expr)} = {format_numeric_value(val)}"]

    try:
        lowered = ql.lower()

        # ---------- 1. Solve equations ----------
        if lowered.startswith("solve") or " solve " in lowered:
            eq_match = re.search(r"solve\s+(.*)\s*=\s*(.*)", ql, flags=re.I)
            if eq_match:
                lhs_text = eq_match.group(1).strip()
                rhs_text = eq_match.group(2).strip()
                expr = safe_sympify(lhs_text) - safe_sympify(rhs_text)
                poly = sp.expand(expr)
                sol = sp.solve(expr, x)

                # Quadratic case
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

        # ---------- 2. Derivative ----------
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

        # ---------- 3. Integration ----------
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

        # ---------- 4. Compute / Evaluate / Substitution ----------
        calc_match = re.match(r'^(compute|calculate|evaluate|what is|find)\s+(.*)', lowered)
        if calc_match:
            expr_text = calc_match.group(2).strip()
            # Handle "at x=..." substitution
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

        # ---------- 5. Plot (optional) ----------
        if re.search(r"\bplot\b", lowered):
            expr_text = re.search(r"plot\s+(.*)", ql, flags=re.I)
            expr_text = expr_text.group(1).strip() if expr_text else "x**2 - 4*x + 3"
            expr = safe_sympify(expr_text)
            xs_plot = np.linspace(-10, 10, 400)
            ys_plot = [float(sp.N(expr.subs(x, xv))) for xv in xs_plot]
            plt.figure(figsize=(6,4))
            plt.plot(xs_plot, ys_plot, label=pretty_power(expr_text))
            plt.title(f"[Wolfram Agent] Plot of {pretty_power(expr_text)}")
            plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.legend()
            plt.show()
            return f"✅ Plot generated for {pretty_power(expr_text)}"

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

# Hugging Face Backup Client
HF_TOKEN = " "
hf_client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=HF_TOKEN)

genai.configure(api_key=API_KEY)

def image_agent(prompt):
    """
    Generate an educational image using:
    1. Google Gemini (primary)
    2. Hugging Face Stable Diffusion XL (backup if Gemini quota fails)
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
                    import base64
                    img_data = part.inline_data.data
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(BytesIO(img_bytes))
                    filename = f"gemini_generated_{idx}.png"
                    img.save(filename)
                    plt.imshow(img)
                    plt.axis("off")
                    plt.show()

                    return f"[Image Agent] ✅ Gemini image generated: {filename}"

    except Exception as e:
        print(f"⚠️ Gemini failed: {e}")

    # ---------- Hugging Face Backup ----------
    try:
        print("🟡 Falling back to Hugging Face Stable Diffusion...")
        image = hf_client.text_to_image(prompt=full_prompt)
        filename = "hf_generated.png"
        image.save(filename)

        plt.imshow(image)
        plt.axis("off")
        plt.show()

        return f"[Image Agent] ✅ Hugging Face image generated: {filename}"
    except Exception as e:
        return f"[Image Agent] ❌ Both Gemini & Hugging Face failed. Error: {e}"


# ============================ #
# Step 12b: Interactive Debug/History Toggle
# ============================ #
DEBUG_MODE = input("Enable debug mode? (True/False): ").strip().lower() == "true"
DEBUG_HISTORY = input("Enable chat history printing? (True/False): ").strip().lower() == "true"
print(f"\n✅ Debug Mode: {DEBUG_MODE}, Chat History: {DEBUG_HISTORY}")

# ============================ #
# Step 13: Chat Loop with Strict Agent Routing
# ============================ #

print("✅ EDUBOT ready! Type 'exit' or 'quit' to stop.\nType '/debug' or '/history' to toggle flags anytime.")

with open("edubot_logs.csv", "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Query", "Answer", "Agent", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "SemanticSim"])

    while True:
        try:
            query = input("\n🟢 Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not query:
            continue
        if query.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break

        # Toggle debug/history
        if query.lower() == "/debug":
            DEBUG_MODE = not DEBUG_MODE
            print(f"🔧 Debug Mode now: {DEBUG_MODE}")
            continue
        if query.lower() == "/history":
            DEBUG_HISTORY = not DEBUG_HISTORY
            print(f"🔧 Chat History Printing now: {DEBUG_HISTORY}")
            continue

        # Greeting
        if query.lower() in ["hi", "hello", "hey"]:
            answer = "Hi there! How can I help you today?"
            agent_type = "Greeting Agent"
            print("\n🟢 Question:\n", query)
            print("\n💬 Answer:\n", answer)
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(answer)
            writer.writerow([query, answer, agent_type, None, None, None, None, None])
            continue

        # Non-academic filter
        if not is_academic_question(query):
            answer = "This question is not related to the study material."
            agent_type = "Non-Academic"
            print("\n🟢 Question:\n", query)
            print("\n💬 Answer:\n", answer)
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(answer)
            writer.writerow([query, answer, agent_type, None, None, None, None, None])
            continue

        # Retrieve context from FAISS
        try:
            docs = retriever.get_relevant_documents(query)
            context_text = " ".join([doc.page_content for doc in docs]) if docs else ""
        except Exception:
            context_text = ""

        if DEBUG_MODE:
            print("\n🔎 [DEBUG] Retrieved Context (first 300 chars):")
            print(context_text[:300] + "..." if context_text else "⚠️ No context retrieved")

        answer = None
        agent_type = None
        query_lower = query.lower()

        # ---------- 1️⃣ Wolfram Agent (math/computation/plot only) ----------
        wolfram_keywords = ["solve", "derivative", "differentiate", "integrate", "integration",
                            "compute", "calculate", "evaluate", "plot", "at x="]
        if any(word in query_lower for word in wolfram_keywords):
            try:
                w = wolfram_agent(query)
                if w:
                    answer = w
                    agent_type = "Wolfram Agent"
            except Exception:
                answer = None

        # ---------- 2️⃣ Quiz / MCQ Agent ----------
        elif "mcq" in query_lower or "quiz" in query_lower:
            try:
                q = quiz_agent(query, context_text)
                if q:
                    answer = q
                    agent_type = "Quiz Agent"
            except Exception:
                answer = None

        # ---------- 3️⃣ Image / Diagram Agent ----------
        elif any(word in query_lower for word in ["sketch", "diagram", "draw", "generate"]):
            try:
                img_resp = image_agent(query)
                if img_resp:
                    answer = img_resp
                    agent_type = "Image Agent"
            except Exception:
                answer = None

        # ---------- 4️⃣ RAG LLM Fallback ----------
        if not answer:
            try:
                result = pdf_chain({"question": query})
                answer = result.get("answer", "").strip() + " [General AI Agent]"
                agent_type = "RAG LLM"
            except Exception as e:
                answer = f"[Error in LLM]: {e}"
                agent_type = "Error"

        # ---------- Print answer ----------
        print("\n🟢 Question:\n", query)
        print("\n💬 Answer:\n", answer)

        # ---------- Debug: Full chat history ----------
        if DEBUG_HISTORY:
            print("\n📚 [DEBUG] Full Chat History:")
            for i, msg in enumerate(memory.load_memory_variables({}).get("chat_history", []), 1):
                msg_type = getattr(msg, "type", "Message")
                content = getattr(msg, "content", str(msg))
                print(f"{i}. {msg_type}: {content}")

        # ---------- Update memory ----------
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(answer)

        # ---------- Evaluation ----------
        reference = context_text if context_text.strip() else None
        eval_scores = evaluate_response(reference, answer)
        rouge_flat = eval_scores["ROUGE"] or {}

        if eval_scores["BLEU"] is not None:
            print("\n📊 Evaluation Scores:")
            print(f"BLEU: {round(eval_scores['BLEU'], 4)}")
            print(f"ROUGE-1: {round(rouge_flat['rouge-1']['f'], 4)}, "
                  f"ROUGE-2: {round(rouge_flat['rouge-2']['f'], 4)}, "
                  f"ROUGE-L: {round(rouge_flat['rouge-l']['f'], 4)}")
            print(f"Semantic Similarity: {eval_scores['SemanticSim']}")
        else:
            print("\n📊 Evaluation Scores: No reference available 🙂")

        # ---------- Write to CSV ----------
        writer.writerow([
            query,
            answer,
            agent_type,
            eval_scores.get("BLEU"),
            rouge_flat.get("rouge-1", {}).get("f", None),
            rouge_flat.get("rouge-2", {}).get("f", None),
            rouge_flat.get("rouge-l", {}).get("f", None),
            eval_scores.get("SemanticSim")
        ])

