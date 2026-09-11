import re
import pandas as pd
from difflib import SequenceMatcher
from typing import List, Tuple, Optional
from dataclasses import dataclass


# Confidence below this counts a word as "matched-but-unclear" even when the
# text lines up correctly. Matches the value used in the hybrid phoneme-aware
# notebook (previously 0.30 in this app).
LOW_CONF = 0.35

CSV_NAME = "evaluation.csv"

# Filler words get a cheaper insertion penalty than substantive extra words.
FILLERS = {
    "uh", "um", "hmm", "like", "youknow", "kinda", "sortof", "sorry",
    "thanks", "thank", "you", "next", "wait", "okay", "ok", "yeah", "yea",
    "nope", "maam", "pardon", "excuseme", "please",
}


def normalize_token(token: str) -> str:
    """Normalize words so alignment is robust to casing and punctuation."""
    token = token.lower().strip()
    token = re.sub(r"[^a-z0-9']+", " ", token)
    token = token.replace("'", "")
    token = re.sub(r"\s+", " ", token).strip()
    return token


def tokenize_reference(text: str) -> List[str]:
    """Split a reference passage into normalized word tokens."""
    normalized = normalize_token(text)
    return normalized.split() if normalized else []


def collapse_repeats(text: str) -> str:
    """Collapse repeated characters in a phonetic key approximation."""
    if not text:
        return text
    out = [text[0]]
    for char in text[1:]:
        if char != out[-1]:
            out.append(char)
    return "".join(out)


def phonetic_key(word: str) -> str:
    """Lightweight, heuristic phonetic proxy for rough pronunciation matching.

    Intentionally simple so it can run without external phoneme libraries.
    """
    text = normalize_token(word).replace(" ", "")
    if not text:
        return ""

    replacements = (
        ("ough", "off"),
        ("augh", "af"),
        ("eigh", "ai"),
        ("igh", "ai"),
        ("tion", "shun"),
        ("sion", "zhun"),
        ("ture", "cher"),
        ("dge", "j"),
        ("tch", "ch"),
        ("ph", "f"),
        ("qu", "kw"),
        ("kn", "n"),
        ("wr", "r"),
        ("wh", "w"),
        ("ck", "k"),
        ("dg", "j"),
    )
    for src, dest in replacements:
        text = text.replace(src, dest)

    text = re.sub(r"c(?=[eiy])", "s", text)
    text = text.replace("c", "k")
    text = re.sub(r"g(?=[eiy])", "j", text)
    text = text.replace("x", "ks")
    text = re.sub("q", "k", text)
    text = text.replace("z", "s")
    text = text.replace("v", "f")

    chars: List[str] = []
    vowels = set("aeiouy")
    for idx, char in enumerate(text):
        if char in vowels:
            # Keep the original first vowel, generalize the rest to 'a'.
            chars.append(char if idx == 0 else "a")
        else:
            chars.append(char)
    return collapse_repeats("".join(chars))


def phonetic_similarity(reference: str, heard: str) -> float:
    """Blend lexical and phonetic similarity into one word-level score."""
    ref_norm = normalize_token(reference)
    heard_norm = normalize_token(heard)
    lexical = SequenceMatcher(None, ref_norm, heard_norm).ratio()
    phonetic = SequenceMatcher(None, phonetic_key(reference), phonetic_key(heard)).ratio()
    prefix_bonus = 0.05 if ref_norm[:1] == heard_norm[:1] else 0.0
    return min(1.0, (0.45 * lexical) + (0.5 * phonetic) + prefix_bonus)


@dataclass
class Word:
    text: str
    start: Optional[float]
    end: Optional[float]
    prob: Optional[float]


def substitution_cost(reference: str, heard: Word) -> Tuple[float, float]:
    """Return weighted alignment cost and similarity for a substitution."""
    similarity = phonetic_similarity(reference, heard.text)
    if similarity < 0.35:
        return 1.25, similarity
    if similarity < 0.55:
        return 0.95, similarity
    return 1.0 - similarity, similarity


def insertion_cost(heard: Word) -> float:
    """Give filler insertions a cheaper penalty than substantive extras."""
    return 0.45 if heard.text in FILLERS else 0.9


def deletion_cost(_: str) -> float:
    """Deletion cost for a missing reference word."""
    return 0.9


def classify_alignment(
    operation: str,
    reference_word: Optional[str],
    heard_word: Optional[Word],
    similarity: Optional[float],
) -> str:
    """Convert an alignment operation into the output evaluation label."""
    if operation == "equal":
        if heard_word and heard_word.prob is not None and heard_word.prob < LOW_CONF:
            return "matched-but-unclear"
        return "match"
    if operation == "sub":
        assert similarity is not None
        if similarity >= 0.88:
            if heard_word and heard_word.prob is not None and heard_word.prob < LOW_CONF:
                return "matched-but-unclear"
            return "match"
        if similarity >= 0.65:
            return "mispronounced"
        return "mispronounced (severe)"
    if operation == "ins":
        if heard_word and heard_word.text in FILLERS:
            return "extra-filler"
        return "extra"
    if operation == "del":
        return "missed"
    raise ValueError(f"Unsupported operation '{operation}'")


def align_reference_to_heard(reference_words: List[str], heard_words: List[Word]) -> List[dict]:
    """Find the best overall alignment between reference and heard words.

    Uses dynamic programming to evaluate three possibilities at each step:
    match/substitute a word, skip a reference word, or insert an extra heard
    word. Substitution cost depends on phonetic and spelling similarity, so
    words that sound close are treated more gently than completely different
    words.
    """
    n = len(reference_words)
    m = len(heard_words)

    scores = [[0.0] * (m + 1) for _ in range(n + 1)]
    ops: List[List[Optional[str]]] = [[None] * (m + 1) for _ in range(n + 1)]
    sims: List[List[Optional[float]]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        scores[i][0] = scores[i - 1][0] + deletion_cost(reference_words[i - 1])
        ops[i][0] = "del"
    for j in range(1, m + 1):
        scores[0][j] = scores[0][j - 1] + insertion_cost(heard_words[j - 1])
        ops[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ref_word = reference_words[i - 1]
            heard_word = heard_words[j - 1]
            sub_cost, similarity = substitution_cost(ref_word, heard_word)
            exact_match = normalize_token(ref_word) == heard_word.text
            operation = "equal" if exact_match else "sub"

            candidates = [
                (scores[i - 1][j] + deletion_cost(ref_word), "del", None),
                (scores[i][j - 1] + insertion_cost(heard_word), "ins", None),
                (scores[i - 1][j - 1] + sub_cost, operation, similarity),
            ]
            best_score, best_op, best_similarity = min(candidates, key=lambda item: item[0])
            scores[i][j] = best_score
            ops[i][j] = best_op
            sims[i][j] = best_similarity

    aligned_rows: List[dict] = []
    i, j = n, m
    while i > 0 or j > 0:
        operation = ops[i][j]
        if operation in ("equal", "sub") and i > 0 and j > 0:
            ref_word = reference_words[i - 1]
            heard_word = heard_words[j - 1]
            similarity = sims[i][j]
            label = classify_alignment(operation, ref_word, heard_word, similarity)
            aligned_rows.append({
                "type": label,
                "reference_word": ref_word,
                "heard_word": heard_word.text,
                "start_sec": None if heard_word.start is None else round(heard_word.start, 3),
                "end_sec": None if heard_word.end is None else round(heard_word.end, 3),
                "confidence": None if heard_word.prob is None else round(heard_word.prob, 3),
            })
            i -= 1
            j -= 1
        elif operation == "del" and i > 0:
            ref_word = reference_words[i - 1]
            label = classify_alignment("del", ref_word, None, None)
            aligned_rows.append({
                "type": label,
                "reference_word": ref_word,
                "heard_word": "",
                "start_sec": None,
                "end_sec": None,
                "confidence": None,
            })
            i -= 1
        elif operation == "ins" and j > 0:
            heard_word = heard_words[j - 1]
            label = classify_alignment("ins", None, heard_word, None)
            aligned_rows.append({
                "type": label,
                "reference_word": "",
                "heard_word": heard_word.text,
                "start_sec": None if heard_word.start is None else round(heard_word.start, 3),
                "end_sec": None if heard_word.end is None else round(heard_word.end, 3),
                "confidence": None if heard_word.prob is None else round(heard_word.prob, 3),
            })
            j -= 1
        else:
            raise RuntimeError(f"Alignment backtrack failed at i={i}, j={j}, op={operation}")

    aligned_rows.reverse()
    return aligned_rows


def evaluate_pronounciations(utterances: List[dict], reference: str) -> pd.DataFrame:
    hyp_words: List[Word] = []
    for utter in utterances:
        utter_words = utter["words"]
        for word_ in utter_words:
            hyp_words.append(Word(
                text=normalize_token(word_["word"].strip()),
                start=word_["start"],
                end=word_["end"],
                prob=word_.get("confidence", None),
            ))

    reference_words = tokenize_reference(reference)
    alignment_rows = align_reference_to_heard(reference_words, hyp_words)

    df = pd.DataFrame(
        alignment_rows,
        columns=["type", "reference_word", "heard_word", "start_sec", "end_sec", "confidence"]
    )

    return df