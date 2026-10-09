import re

SPELLING = [("execuition", "execution"), ("mislaneous", "miscellaneous"), ("misclaneous", "miscellaneous"),
            ("maintainance", "maintenance"), ("partion", "partition")]


def key(code):
    """Same matching rule as the case-type cleanup in scripts/pipeline/normalize.py."""
    t = re.sub(r"[^a-z0-9]", "", str(code).lower())
    for wrong, right in SPELLING:
        t = t.replace(wrong, right)
    t = re.sub(r"complain(?!t)", "complaint", t)
    return re.sub(r"(cases|case)$", "", t) or t


def _build(pairs):
    return {key(c): label for c, label in pairs}


BASE = [
    ("other", "Other case types (grouped together)"),
    ("gr case", "G.R. case (police case, General Register)"),
    ("cs", "Civil Suit"), ("civil suit", "Civil Suit"),
    ("mact", "Motor Accident Claims Tribunal case (MACT)"),
    ("mac case", "Motor Accident Claims case (MAC)"),
    ("session case", "Sessions Case"),
    ("execution cases", "Execution Case"),
    ("probate case", "Probate Case"),
    ("title suit", "Title Suit"), ("title appeal", "Title Appeal"),
    ("cr rev", "Criminal Revision"), ("cri rev", "Criminal Revision"),
    ("matrimonial case", "Matrimonial Case"),
    ("miscellaneous", "Miscellaneous Case"), ("misc cases", "Miscellaneous Case"),
    ("complaint", "Complaint Case"),
    ("juvenile case", "Juvenile Case"),
    ("money suit", "Money Suit"),
    ("partition suit", "Partition Suit"),
    ("civil misc case", "Civil Miscellaneous Case"),
    ("misc civil appeal", "Miscellaneous Civil Appeal"),
    ("maintenance", "Maintenance Case"),
    ("divorce", "Divorce Case"),
    ("succession", "Succession Case"),
    ("special case", "Special Case"),
    ("claim cases", "Claim Case (unconfirmed)"),
]

ODISHA = BASE + [
    ("crlt", "Criminal Trial (CRLT, unconfirmed)"),
    ("ui", "U.I. case (unconfirmed)"),
    ("1cc", "Complaint Case, class 1 (unconfirmed)"),
    ("1(c)cc", "Complaint Case, class 1(c) (unconfirmed)"),
    ("2(a)cc", "Complaint Case, class 2(a) (unconfirmed)"),
    ("2(b)cc", "Complaint Case, class 2(b) (unconfirmed)"),
    ("2(c)cc", "Complaint Case, class 2(c) (unconfirmed)"),
    ("icc", "I.C.C. case (unconfirmed)"),
    ("uc", "U.C. case (unconfirmed)"),
    ("cs(i)", "Civil Suit, class I (unconfirmed)"),
    ("cs (iii)", "Civil Suit, class III (unconfirmed)"),
    ("st", "Sessions Trial (ST)"),
    ("ia", "Interlocutory Application (IA)"),
    ("rfa", "Regular First Appeal (RFA)"),
    ("exp", "Execution Proceeding (EXP, unconfirmed)"),
    ("blapl", "Bail Application (BLAPL)"),
    ("crlmc", "Criminal Miscellaneous Case (CRLMC)"),
    ("cma", "Civil Miscellaneous Appeal (CMA)"),
    ("fao", "First Appeal from Order (FAO)"),
    ("cr case complaint (p)", "Criminal Complaint Case (P, unconfirmed)"),
    ("ct", "Court case (CT, unconfirmed)"),
    ("crl tr", "Criminal Trial"),
    ("cri bail appln", "Criminal Bail Application"),
    ("un important case", "Unimportant case (as recorded)"),
    ("laref", "Land Acquisition Reference (LAREF)"),
    ("spl g r", "Special G.R. case (unconfirmed)"),
    ("jgr", "Juvenile G.R. case (unconfirmed)"),
    ("tr case", "Trial case (unconfirmed)"),
    ("mat case", "Matrimonial Case (MAT)"),
    ("crl misc case", "Criminal Miscellaneous Case"),
    ("ts", "Title Suit (TS)"),
    ("test case", "Test case (as recorded)"),
]

BIHAR = BASE + [
    ("cri case", "Criminal Case"), ("cr case", "Criminal Case"), ("criminal case", "Criminal Case"),
    ("reg cri case", "Regular Criminal Case"),
    ("cr case complaint (p)", "Criminal Complaint Case (P, unconfirmed)"),
    ("cr case complaint (o)", "Criminal Complaint Case (O, unconfirmed)"),
    ("crcase complt(p)", "Criminal Complaint Case (P, unconfirmed)"),
    ("anticipatory bail", "Anticipatory Bail"),
    ("regular bail", "Regular Bail"),
    ("abp", "Anticipatory Bail Petition (ABP, unconfirmed)"),
    ("bp", "Bail Petition (BP, unconfirmed)"),
    ("indian penal code (ipc)", "Indian Penal Code (IPC) case"),
    ("gr police cases", "G.R. police case"),
    ("general register", "General Register case"),
    ("cri rev app", "Criminal Revision Application"),
    ("cri appeal", "Criminal Appeal"),
    ("excise act", "Excise Act case"),
    ("spl case", "Special Case"),
    ("misc cases a) o-9 , r-4 b) o-9, r-9 c) u/s 47cpc", "Miscellaneous civil case (Order 9 / Section 47 CPC)"),
    ("criminal case or gr", "Criminal case or G.R. case"),
    ("cri case (gr)", "Criminal case (G.R.)"),
    ("special case (sc / st act)", "Special case (SC/ST Act)"),
    ("ndps s case", "NDPS special case (unconfirmed)"),
    ("mv claim cases", "Motor Vehicle claim case"),
]

TABLES = {"odisha": _build(ODISHA), "bihar": _build(BIHAR)}


def has_label(state, code):
    return key(code) in TABLES.get(state, {})


def case_type_label(state, code):
    label = TABLES.get(state, {}).get(key(code))
    if label:
        return label
    return f"{str(code).upper()} (court code, meaning not confirmed)"


def gender_label(code):
    c = str(code).strip().lower()
    if c in ("missing", "nan", ""):
        return "Not recorded"
    try:
        n = int(float(c))
    except ValueError:
        n = None
    if "-9998" in c or "unclear" in c or n == -9998:
        return "Unclear / not classified"
    if "-9999" in c or "missing" in c or n == -9999:
        return "Name missing in record"
    if "female" in c or n == 1:
        return "Female"
    if "male" in c or n == 0:
        return "Male"
    return str(code)
