import re


def score_output(generated_text: str, expected_answer: str, difficulty: str) -> int:
    """
    Evaluates whether the model's generated text matches the expected answer.

    - Easy: Checks that the expected answer string appears in the lowercased output.
    - Hard (GSM8K): Extracts the last number in the model's reasoning and compares numerically.

    Returns: 1 if correct, 0 if incorrect.
    """
    gen_clean = generated_text.strip().lower()
    expected_clean = expected_answer.strip().lower()

    if difficulty == "easy":
        return 1 if expected_clean in gen_clean else 0
    else:
        # Remove commas in numbers like "18,000" -> "18000"
        cleaned = generated_text.replace(",", "")
        numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        if not numbers:
            return 0
        last_num = numbers[-1]
        try:
            return 1 if float(last_num) == float(expected_clean) else 0
        except ValueError:
            return 1 if last_num == expected_clean else 0
