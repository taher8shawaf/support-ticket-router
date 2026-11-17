"""
Area B - Classical NLP utilities.

  * Regex text normalisation (Turkish-aware lower-casing, URL / e-mail masking,
    punctuation stripping, elongation squashing).
  * Regex model-code / error-code extraction (e.g. "DW-6534", "E04").
  * Minimum-edit-distance (Levenshtein, dynamic programming) spelling corrector
    for appliance vocabulary in English and Turkish.
  * ELIZA-style regex rules that answer greetings / small talk instantly,
    before any model inference.

`preprocess(text)` is THE single preprocessing entry point. Every trainer,
evaluate.py and app.py call it, so all four models always see exactly the
same input string (`PreprocessResult.model_input`).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 1. Regex normalisation
# --------------------------------------------------------------------------- #
# Turkish -> ASCII folding. Strictly one character to one character, so string
# indices stay aligned between the folded and the unfolded text.
_TR_FOLD = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")

# Model numbers: 2-4 letters, optional hyphen, 3-5 digits, optional 0-2 letter
# suffix. Matches DW-6534, RF-1180X, WM8000, AC-9000.
MODEL_CODE_RE = re.compile(r"\b([A-Za-z]{2,4})-?(\d{3,5})([A-Za-z]{0,2})\b")
# Error codes shown on appliance displays: E04, F21, H-2, e 15 is NOT matched
# (space) to avoid false positives on words like "e 15".
ERROR_CODE_RE = re.compile(r"\b([EeFfHh])-?(\d{1,3})\b")

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
NON_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
ELONGATION_RE = re.compile(r"(\w)\1{2,}", flags=re.UNICODE)   # "nooooo" -> "noo"
WS_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

MODEL_PLACEHOLDER = "modelcode"
ERROR_PLACEHOLDER = "errorcode"


def fold_turkish(text: str) -> str:
    """'bulaşık' -> 'bulasik' (1:1 character mapping)."""
    return text.translate(_TR_FOLD)


def turkish_safe_lower(text: str) -> str:
    """Lower-case that maps 'İ' to 'i' (Python's default produces 'i̇').

    Plain 'I' is mapped to 'i' (not dotless 'ı') because the system is
    bilingual and English "I" is far more frequent in support messages.
    """
    return text.replace("İ", "i").lower()


def extract_codes(text: str) -> tuple[list[str], list[str]]:
    """Return (model_codes, error_codes) in canonical upper-case form."""
    model_codes = []
    for m in MODEL_CODE_RE.finditer(text):
        code = f"{m.group(1).upper()}-{m.group(2)}{m.group(3).upper()}"
        if code not in model_codes:
            model_codes.append(code)
    error_codes = []
    for m in ERROR_CODE_RE.finditer(text):
        code = f"{m.group(1).upper()}{int(m.group(2)):02d}"
        if code not in error_codes:
            error_codes.append(code)
    return model_codes, error_codes


def normalize(text: str) -> str:
    """Regex normalisation pipeline (codes are masked with placeholders)."""
    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub(" url ", text)
    text = EMAIL_RE.sub(" email ", text)
    # Mask codes BEFORE lower-casing/punctuation removal so the hyphen survives.
    text = MODEL_CODE_RE.sub(f" {MODEL_PLACEHOLDER} ", text)
    text = ERROR_CODE_RE.sub(f" {ERROR_PLACEHOLDER} ", text)
    text = turkish_safe_lower(text)
    text = text.replace("'", " ").replace("’", " ")
    text = NON_WORD_RE.sub(" ", text)
    text = text.replace("_", " ")
    text = ELONGATION_RE.sub(r"\1\1", text)
    return WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


# --------------------------------------------------------------------------- #
# 2. Minimum edit distance spelling correction
# --------------------------------------------------------------------------- #
def levenshtein(a: str, b: str) -> int:
    """Minimum edit distance (insert = delete = substitute = 1), O(|a|*|b|) DP.

    D[i][j] = min(D[i-1][j] + 1,          # deletion
                  D[i][j-1] + 1,          # insertion
                  D[i-1][j-1] + (a_i != b_j))  # substitution / match
    Only two rows are kept in memory.
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1,              # deletion
                            curr[j - 1] + 1,          # insertion
                            prev[j - 1] + (ca != cb)))  # substitution
        prev = curr
    return prev[-1]


# Canonical appliance / support vocabulary (EN + TR, incl. common inflections
# so correct inflected words are never "corrected" into another form).
APPLIANCE_LEXICON = [
    # English - products
    "washing", "machine", "washer", "dishwasher", "dryer", "tumble",
    "refrigerator", "fridge", "freezer", "oven", "microwave", "vacuum",
    "cleaner", "conditioner", "stove", "cooktop", "kettle", "appliance",
    "appliances", "thermostat", "compressor", "detergent", "filter",
    # English - problems / support words
    "error", "leaking", "leaks", "draining", "drain", "cooling", "heating",
    "noise", "noisy", "spinning", "broken", "working", "display", "door",
    "technician", "appointment", "warranty", "repair", "installation",
    "delivery", "refund", "replacement", "notification", "firmware",
    "connect", "wifi", "temperature",
    # Turkish - products
    "çamaşır", "bulaşık", "makine", "makina", "makinesi", "makinem",
    "makinemi", "makinemin", "makinesinin", "makineme", "buzdolabı",
    "buzdolabım", "buzdolabımı", "buzdolabının", "dondurucu", "fırın",
    "fırınım", "mikrodalga", "kurutma", "süpürge", "süpürgesi", "klima",
    "klimam", "klimamın", "ocak", "çaydanlık", "kahve", "deterjan", "filtre",
    # Turkish - problems / support words
    "hata", "hatası", "kodu", "arıza", "arızalı", "bozuk", "boşaltmıyor",
    "soğutmuyor", "ısınmıyor", "ısıtmıyor", "çalışmıyor", "sızdırıyor",
    "sızıyor", "akıtıyor", "kapağı", "ekran", "ekranında", "teknisyen",
    "servis", "randevu", "randevusu", "randevumu", "garanti", "tamir",
    "kurulum", "teslimat", "teslimatım", "sipariş", "siparişim", "iade",
    "değişim", "uygulama", "bildirim", "bağlanmıyor",
    # Positive verb forms MUST be in the lexicon too: otherwise "çalışıyor"
    # (it works) would be one edit away from "çalışmıyor" (it does not work)
    # and the corrector would flip the meaning of the sentence.
    "çalışıyor", "boşaltıyor", "soğutuyor", "ısınıyor", "ısıtıyor",
    "sızdırmıyor", "bağlanıyor", "akıtmıyor",
]
_LEXICON_SET = set(APPLIANCE_LEXICON)
_FOLDED_LEXICON: dict[str, str] = {}
for _w in APPLIANCE_LEXICON:
    _FOLDED_LEXICON.setdefault(fold_turkish(_w), _w)

# Frequent words that are 1 edit away from a lexicon entry but are correct
# words in their own right - never touch them.
PROTECTED_WORDS = {
    "over", "even", "open", "opens", "dryers", "doors", "water", "after",
    "rather", "matter", "machines", "washers", "fridges", "ovens", "errors",
    "filters", "drains", "leak", "leaked", "worked", "works", "connected",
    "makes", "making", "cleaning", "cleaned", "heater", "cooler", "kodum",
    "kodunu", "hatam", "servise", "servisi", "ocağı", "kahvem", "klimayı",
    "fırını", "fırına", "garantisi", "siparişi", "randevum", "servisim",
}


@dataclass
class Correction:
    original: str
    corrected: str
    distance: int


def correct_token(token: str) -> tuple[str, int] | None:
    """Return (corrected_word, edit_distance) or None if no correction.

    Rules (deliberately conservative to avoid over-correction):
      1. Exact lexicon words, protected words, numbers, placeholders: keep.
      2. Diacritic-less Turkish ("bulasik") -> exact folded match (distance 0
         after folding, reported as the raw edit distance).
      3. Otherwise words with >= 5 characters may be corrected to the closest
         lexicon entry with the same first (folded) letter:
         max distance 1 for 5-7 chars, 2 for >= 8 chars.
    """
    if (len(token) < 3 or token in _LEXICON_SET or token in PROTECTED_WORDS
            or token.isdigit() or token in (MODEL_PLACEHOLDER, ERROR_PLACEHOLDER)):
        return None
    folded = fold_turkish(token)
    if folded in _FOLDED_LEXICON:
        target = _FOLDED_LEXICON[folded]
        return (target, levenshtein(token, target)) if target != token else None
    if len(token) < 5:
        return None
    max_dist = 1 if len(token) <= 7 else 2
    best, best_key = None, None
    for f_word, word in _FOLDED_LEXICON.items():
        if f_word[0] != folded[0] or abs(len(f_word) - len(folded)) > max_dist:
            continue
        d = levenshtein(folded, f_word)
        if d <= max_dist:
            key = (d, abs(len(f_word) - len(folded)))
            if best_key is None or key < best_key:
                best, best_key = word, key
    if best is None:
        return None
    return best, levenshtein(token, best)


def spell_correct(text: str) -> tuple[str, list[Correction]]:
    """Correct every token of an already-normalised text."""
    out, corrections = [], []
    for tok in text.split():
        res = correct_token(tok)
        if res is None:
            out.append(tok)
        else:
            out.append(res[0])
            corrections.append(Correction(tok, res[0], res[1]))
    return " ".join(out), corrections


# --------------------------------------------------------------------------- #
# 3. ELIZA-style small-talk rules
# --------------------------------------------------------------------------- #
# Patterns are matched against the *folded* (ASCII) lower-case text, so they
# work with and without Turkish characters. They are anchored (^...$): only a
# message that is ENTIRELY small talk is answered; "hi, my oven is broken"
# goes to the models. {0}, {1} insert captured groups (ELIZA-style).
_ELIZA_RULES: list[tuple[re.Pattern, str, str]] = [
    # ---- English ----
    (r"^(hi|hello|hey|hiya|howdy|good (morning|afternoon|evening))( there)?( team| support| everyone)?$",
     "Hello! 👋 I'm the support assistant. Please describe the problem with your appliance "
     "(you can include the model number, e.g. DW-6534).", "en"),
    (r"^(thanks|thank you|thank you very much|thanks a lot|thx|ty|cheers)( so much)?$",
     "You're welcome! Is there anything else I can help you with?", "en"),
    (r"^how are you( doing)?( today)?$",
     "I'm doing well, thank you for asking! How can I help with your appliance today?", "en"),
    (r"^(bye|goodbye|bye bye|see you|see you later|have a nice day)$",
     "Goodbye! Have a great day. 👋", "en"),
    (r"^(who|what) are you$|^are you (a )?(bot|robot|human|real person)$",
     "I'm an automated assistant that routes your message to the right support team.", "en"),
    (r"^my name is (\w+)$",
     "Nice to meet you, {0}! How can I help you with your appliance today?", "en"),
    (r"^i(?: am| m) (happy|sad|angry|upset|bored|tired|fine|good|great|frustrated)$",
     "Why do you feel {0}? If something is wrong with one of your appliances, tell me about it.", "en"),
    (r"^(ok|okay|cool|great|nice|alright|got it)$",
     "Great! Let me know if you need help with an appliance.", "en"),
    # ---- Turkish (written in folded ASCII form) ----
    (r"^(merhaba|merhabalar|selam|selamlar|selamun aleykum|gunaydin|iyi gunler|iyi aksamlar|hey)$",
     "Merhaba! 👋 Ben destek asistanıyım. Lütfen cihazınızdaki sorunu yazın "
     "(model numarasını da ekleyebilirsiniz, örn. DW-6534).", "tr"),
    (r"^(merhaba|selam) (nasilsin|nasilsiniz)$",
     "Merhaba, iyiyim teşekkürler! Cihazınızla ilgili nasıl yardımcı olabilirim?", "tr"),
    (r"^(tesekkurler|tesekkur ederim|cok tesekkurler|cok tesekkur ederim|sag ol|sagol|sagolun|eyvallah)$",
     "Rica ederim! Başka bir konuda yardımcı olabilir miyim?", "tr"),
    (r"^(nasilsin|nasilsiniz|naber|ne haber|iyi misin)$",
     "İyiyim, sorduğunuz için teşekkürler! Cihazınızla ilgili nasıl yardımcı olabilirim?", "tr"),
    (r"^(gorusuruz|hosca kal|hoscakal|hosca kalin|gule gule|iyi geceler)$",
     "Görüşmek üzere, iyi günler! 👋", "tr"),
    (r"^(sen kimsin|kimsin|bot musun|robot musun|insan misin)$",
     "Ben mesajınızı doğru destek ekibine yönlendiren otomatik bir asistanım.", "tr"),
    (r"^(benim )?adim (\w+)$",
     "Memnun oldum, {1}! Cihazınızla ilgili nasıl yardımcı olabilirim?", "tr"),
    (r"^(tamam|peki|olur|super|harika|anladim)$",
     "Harika! Bir cihaz sorununuz olursa yazmanız yeterli.", "tr"),
]
ELIZA_RULES = [(re.compile(p), r, lang) for p, r, lang in _ELIZA_RULES]


@dataclass
class SmallTalkReply:
    reply: str
    lang: str
    pattern: str


def check_small_talk(text: str) -> SmallTalkReply | None:
    """Return an instant rule-based reply if the message is pure small talk."""
    clean = turkish_safe_lower(unicodedata.normalize("NFC", text))
    clean = NON_WORD_RE.sub(" ", clean.replace("'", " ").replace("’", " "))
    clean = ELONGATION_RE.sub(r"\1", WS_RE.sub(" ", clean).strip())
    folded = fold_turkish(clean)          # same length as `clean`
    for pattern, template, lang in ELIZA_RULES:
        m = pattern.match(folded)
        if m:
            # Take captured groups from the UNfolded text (indices align 1:1).
            groups = [clean[m.start(i + 1):m.end(i + 1)].capitalize() if m.group(i + 1) else ""
                      for i in range(len(m.groups()))]
            return SmallTalkReply(template.format(*groups), lang, pattern.pattern)
    return None


# --------------------------------------------------------------------------- #
# 4. Full preprocessing pipeline (shared by ALL models, training + inference)
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessResult:
    original: str
    normalized: str
    corrected: str
    model_codes: list[str] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)

    @property
    def model_input(self) -> str:
        """The exact string fed to Model 1-4."""
        return self.corrected


def preprocess(text: str) -> PreprocessResult:
    model_codes, error_codes = extract_codes(text)
    normalized = normalize(text)
    corrected, corrections = spell_correct(normalized)
    return PreprocessResult(text, normalized, corrected, model_codes,
                            error_codes, corrections)


def preprocess_batch(texts: list[str]) -> list[str]:
    """Convenience: model inputs for a list of raw texts."""
    return [preprocess(t).model_input for t in texts]


if __name__ == "__main__":  # quick manual check: python -m src.nlp_utils
    for s in ["bulasik makinem su boşaltmıyor, model DW-6534",
              "My washng machne shows error E04!!!",
              "refridgerator RF-1180X is leakingg", "Merhaba!", "hi there",
              "my name is Ayşe", "Benim adım Çağrı", "what's the weather tomorrow?"]:
        st = check_small_talk(s)
        if st:
            print(f"{s!r:50} -> SMALL TALK: {st.reply}")
            continue
        r = preprocess(s)
        print(f"{s!r:50} -> {r.model_input!r} codes={r.model_codes} errors={r.error_codes} "
              f"fixes={[(c.original, c.corrected, c.distance) for c in r.corrections]}")
