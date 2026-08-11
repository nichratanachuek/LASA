"""
POCA-style Drug Name Similarity Score
=======================================
Reimplementation of the PUBLISHED methodology behind FDA's Phonetic and
Orthographic Computer Analysis (POCA) tool. This is NOT FDA's actual
software -- POCA's source code is only distributed by request to FDA
(pocasourcecoderequest@fda.hhs.gov), and the hosted tool requires FDA's
internal system (https://poca-public.fda.gov). Use this as a reasonable
approximation for triage, not a substitute for running the real tool.

Per FDA/ISMP documentation, POCA computes:
    score = mean( orthographic_score, phonetic_score )   scaled 0-100

Where:
    orthographic_score = mean( BI-SIM, LED )
        BI-SIM = bigram-alignment similarity ("look-alike")
        LED    = normalized edit distance ("look-alike")
    phonetic_score = ALINE phonetic-alignment score ("sound-alike")
        (ALINE is a complex articulatory-feature alignment algorithm;
        this implementation substitutes a Double Metaphone code-match
        score, which is a simpler but reasonable phonetic proxy)

Real POCA scores >= 70 are labeled "Highly Similar" by FDA.
"""

import re


# ----------------------------------------------------------------------
# Orthographic component 1: LED (normalized Levenshtein edit distance)
# ----------------------------------------------------------------------
def levenshtein_distance(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def led_score(a: str, b: str) -> float:
    """Normalized edit distance similarity, 0-1 (LED proxy)."""
    if not a and not b:
        return 1.0
    dist = levenshtein_distance(a, b)
    return 1 - dist / max(len(a), len(b))


# ----------------------------------------------------------------------
# Orthographic component 2: BI-SIM (bigram similarity)
# ----------------------------------------------------------------------
def get_bigrams(s: str) -> set:
    s = s.lower()
    if len(s) < 2:
        return {s}
    return {s[i:i + 2] for i in range(len(s) - 1)}


def bisim_score(a: str, b: str) -> float:
    """Dice coefficient over bigrams, 0-1 (BI-SIM proxy)."""
    set_a, set_b = get_bigrams(a), get_bigrams(b)
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    return 2 * intersection / (len(set_a) + len(set_b))


# ----------------------------------------------------------------------
# Phonetic component: Double Metaphone code match (ALINE proxy)
# ----------------------------------------------------------------------
def double_metaphone_primary(s: str) -> str:
    """Compact primary-code Double Metaphone. Simplified but captures
    most consonant-sound groupings relevant to drug-name confusion."""
    s = re.sub(r'[^A-Z]', '', s.upper())
    if not s:
        return ""
    vowels = set("AEIOU")
    length = len(s)
    padded = s + "     "
    pos = 0
    code = []

    def at(i):
        return padded[i] if 0 <= i < len(padded) else ""

    if s[0:2] in ("GN", "KN", "PN", "WR", "PS"):
        pos = 1
    elif s[0:1] == "X":
        code.append("S")
        pos = 1

    while pos < length and len(code) < 8:
        c = at(pos)
        if c in vowels:
            if pos == 0:
                code.append("A")
            pos += 1
        elif c == "B":
            code.append("P"); pos += 2 if at(pos + 1) == "B" else 1
        elif c == "C":
            if at(pos + 1) == "H":
                code.append("X"); pos += 2
            elif at(pos + 1) in "IEY":
                code.append("S"); pos += 2
            else:
                code.append("K"); pos += 2 if at(pos + 1) == "C" else 1
        elif c == "D":
            if at(pos + 1) == "G" and at(pos + 2) in "IEY":
                code.append("J"); pos += 3
            else:
                code.append("T"); pos += 2 if at(pos + 1) == "D" else 1
        elif c == "G":
            if at(pos + 1) == "H":
                code.append("K"); pos += 2
            elif at(pos + 1) in "IEY":
                code.append("J"); pos += 2
            else:
                code.append("K"); pos += 2 if at(pos + 1) == "G" else 1
        elif c == "H":
            if at(pos - 1) in vowels and at(pos + 1) not in vowels:
                pos += 1
            else:
                code.append("H"); pos += 1
        elif c == "P":
            if at(pos + 1) == "H":
                code.append("F"); pos += 2
            else:
                code.append("P"); pos += 2 if at(pos + 1) == "P" else 1
        elif c == "Q":
            code.append("K"); pos += 1
        elif c == "S":
            if at(pos + 1) == "H":
                code.append("X"); pos += 2
            else:
                code.append("S"); pos += 2 if at(pos + 1) == "S" else 1
        elif c == "T":
            if at(pos + 1) == "H":
                code.append("0"); pos += 2
            else:
                code.append("T"); pos += 2 if at(pos + 1) == "T" else 1
        elif c == "V":
            code.append("F"); pos += 2 if at(pos + 1) == "V" else 1
        elif c == "X":
            code.append("KS"); pos += 1
        elif c == "Z":
            code.append("S"); pos += 2 if at(pos + 1) == "Z" else 1
        elif c in "JLMNRWY":
            code.append(c); pos += 1
        else:
            pos += 1

    return "".join(code)[:8]


def aline_proxy_score(a: str, b: str) -> float:
    """Phonetic similarity via Double Metaphone code edit-distance, 0-1
    (proxy for POCA's ALINE algorithm)."""
    code_a = double_metaphone_primary(a)
    code_b = double_metaphone_primary(b)
    if not code_a and not code_b:
        return 1.0
    if code_a == code_b:
        return 1.0
    dist = levenshtein_distance(code_a, code_b)
    return 1 - dist / max(len(code_a), len(code_b), 1)


# ----------------------------------------------------------------------
# Combined POCA-style score
# ----------------------------------------------------------------------
def poca_style_score(drug_a: str, drug_b: str) -> dict:
    """
    Returns a dict with the orthographic sub-score, phonetic sub-score,
    and combined POCA-style score, each on a 0-100 scale (matching the
    scale FDA's real POCA tool reports).
    """
    bi_sim = bisim_score(drug_a, drug_b)
    led = led_score(drug_a, drug_b)
    orthographic = (bi_sim + led) / 2 * 100

    aline = aline_proxy_score(drug_a, drug_b)
    phonetic = aline * 100

    combined = (orthographic + phonetic) / 2

    return {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "bi_sim": round(bi_sim * 100, 2),
        "led": round(led * 100, 2),
        "orthographic_score": round(orthographic, 2),
        "aline_proxy": round(aline * 100, 2),
        "phonetic_score": round(phonetic, 2),
        "poca_style_score": round(combined, 2),
        "highly_similar": combined >= 70,   # FDA's published "Highly Similar" cutoff
    }


if __name__ == "__main__":
    examples = [
        ("hydroxyzine", "hydralazine"),
        ("Xarelto 15 mg", "Xarelto 10 mg"),
        ("nifedipine", "manidipine"),
        ("Losec", "Lasix"),
    ]
    for a, b in examples:
        r = poca_style_score(a, b)
        print(f"{a!r:22s} vs {b!r:22s} -> "
              f"orthographic={r['orthographic_score']:5.1f}  "
              f"phonetic={r['phonetic_score']:5.1f}  "
              f"POCA-style={r['poca_style_score']:5.1f}  "
              f"{'HIGHLY SIMILAR' if r['highly_similar'] else ''}")
