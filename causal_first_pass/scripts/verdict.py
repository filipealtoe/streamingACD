import re
RULES = [  # first match wins
    ("misleading or partly false", r"mostly false|partly|partially|half[- ]true|mixture|mixed|misleading|missing context|lacks context|needs context|out of context|exaggerat|overstat|cherry|distort|spin|two pinocchios|three pinocchios|unproven|unsupported|unsubstantiated|not the whole story|half|enganoso|enganador|miscaption|doubtful|selective|outdated|disputed|flawed|yes, but|misconstrued|old photo|no relation"),
    ("false", r"false|falso|errado|fake|pants on fire|incorrect|wrong|fabricat|hoax|not true|baseless|no evidence|untrue|four pinocchios|bogus|scam|altered|doctored|manipulated|debunk|inaccurate|misattributed|fiction|not legit|^no\b"),
    ("true", r"\btrue\b|\bcorrect|\baccurate|\blegit|\bcerto\b|\bverdadeiro"),
    ("satire", r"satire|parody|joke|sarcas"),
]
def group(rating):
    r = (rating or "").lower().strip()
    for name, pat in RULES:
        if re.search(pat, r): return name
    return "other or unrated"
