import re

HARD_WORDS = [
    "prove",
    "calculate",
    "step by step",
    "why does",
    "compare",
    "debug",
    "derive",
    "how many",
    "solve",
    "explain why",
]


def classify(q: str) -> str:
    """
    Rule-based prompt complexity classifier.
    Assigns points based on keywords, numbers, symbols, and length.
    Returns: 'easy', 'medium', or 'hard'.
    """
    ql = q.lower()
    score = 0

    # 1. Keywords that indicate multi-step reasoning (+3)
    if any(w in ql for w in HARD_WORDS):
        score += 3

    # 2. Three or more numbers (+2)
    if len(re.findall(r"\d+", q)) >= 3:
        score += 2

    # 3. Math operators or code syntax (+2)
    if re.search(r"[+\-*/=^]|def |\{|\}", q):
        score += 2

    # 4. Long queries over 40 words (+1)
    if len(q.split()) > 40:
        score += 1

    # 5. Multiple clauses with commas and conjunctions (+1)
    if q.count(",") + q.count(" and ") >= 3:
        score += 1

    return "easy" if score <= 2 else ("medium" if score <= 4 else "hard")
