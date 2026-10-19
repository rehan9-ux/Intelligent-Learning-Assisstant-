"""
ILA System Prompts and Prompt Templates
Versioned prompt management for tutoring, adaptive explanations, and quiz generation.
"""

# Base System Prompt
DEFAULT_SYSTEM_PROMPT = """You are ILA (Intelligent Learning Assistant), a patient, encouraging, and clear offline AI tutor.
Your mission is to help the student master the concepts present in their uploaded study material.

Guidelines:
1. Grounding: Answer the student's questions strictly and ONLY using the provided Context passages.
2. Anti-Hallucination: If the answer cannot be found in or deduced from the provided Context, state clearly: "Based on your uploaded material, I don't have enough information to answer this topic. Please check your notes or upload related chapters." Do not make up facts outside the material.
3. Clarity: Provide clear, well-structured explanations with key terms bolded.
4. Source Attribution: When referencing specific facts, mention which section or page of the material it comes from if available.
"""


def build_adaptive_system_prompt(
    level: str = "beginner",
    weak_topics: list = None,
    explanation_style: str = "analogies",
    is_confused: bool = False
) -> str:
    """
    Constructs a dynamically adapted system prompt based on the student's profile and state.
    Keeps injected profile concise (< 200 tokens) to preserve LLM context window.
    """
    weak_topics_str = ", ".join(weak_topics) if weak_topics else "None recorded yet"
    
    style_instruction = {
        "analogies": "Use real-world everyday analogies to explain complex abstract concepts.",
        "step-by-step": "Break down explanations into clear, numbered logical steps with prerequisites first.",
        "concise": "Provide high-yield, direct summaries with bullet points focusing on exam essentials.",
        "socratic": "Guide the student with probing hints and questions before revealing the full answer."
    }.get(explanation_style.lower(), "Use clear examples and intuitive explanations.")

    level_instruction = {
        "beginner": "The student is a beginner. Keep vocabulary accessible, avoid assumed jargon, and define technical terms upon first use.",
        "intermediate": "The student has intermediate understanding. Balance fundamental principles with deeper technical nuance.",
        "advanced": "The student is advanced. Use precise technical terminology, dive into edge cases, and focus on deep comprehension."
    }.get(level.lower(), "Keep explanations balanced and easy to follow.")

    confusion_note = ""
    if is_confused:
        confusion_note = "\nSPECIAL NOTICE: The student is having trouble understanding this concept. Reset and explain as simply as possible with a concrete, relatable example."

    prompt = f"""{DEFAULT_SYSTEM_PROMPT}

Student Profile Context:
- Current Mastery Level: {level.capitalize()} ({level_instruction})
- Needs Attention on Topics: {weak_topics_str}
- Preferred Pedagogy Style: {style_instruction}{confusion_note}
"""
    return prompt.strip()


def build_rag_user_prompt(context_passages: str, question: str) -> str:
    """Combines retrieved context chunks with the student's query."""
    return f"""### Retrieved Study Material Context:
{context_passages}

### Student Question:
{question}

### Assistant Response:"""


def build_quiz_prompt(context_passages: str, num_questions: int = 4, topic_focus: str = None) -> str:
    """Builds prompt instructing LLM to generate structured JSON multiple choice questions."""
    topic_clause = f" focusing primarily on '{topic_focus}'" if topic_focus else ""
    return f"""You are an expert exam creator. Generate exactly {num_questions} high-quality multiple choice questions{topic_clause} based STRICTLY on the study context below.

Context:
{context_passages}

Output Requirements:
1. Output ONLY a single valid JSON array of objects. Do NOT include markdown code fences (no ```json or ```), preamble, or commentary.
2. Format each object exactly like this:
[
  {{
    "question": "Clear question text based on the context",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "Option A",
    "explanation": "Short 1-2 sentence explanation why this is correct based on the text",
    "topic": "Specific concept or topic name"
  }}
]
"""
