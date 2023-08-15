
"""
    This mapping is used to define which atoms have a relativ vector to each other.
    This is a sorted list of tuples where the first element is the source atom name and the second element is the target atom name.
"""
DOPC_AT_MAPPING = [
    ("N",   "C15"),   # N is the start
    ("N",   "C14"),
    ("N",   "C13"),
    ("N",   "C12"),
    ("C12", "C11"),
    ("C11", "O12"),
    ("P",   "O14"),
    ("P",   "O13"),
    ("P",   "O12"),
    ("P",   "O11"),
    ("O11", "C1"),
    ("C1",  "C2"),   # C2 is the atom where the two arms split
    # Arm 1
    ("C2",   "C3"),
    ("C3",   "O31"),
    ("O31",  "C31"),
    ("C31",  "O32"),  # Small O arm
    ("C31",  "C32"),
    ("C32",  "C33"),
    ("C33",  "C34"),
    ("C34",  "C35"),
    ("C35",  "C36"),
    ("C36",  "C37"),
    ("C37",  "C38"),
    ("C38",  "C39"),
    ("C39",  "C310"),
    ("C310", "C311"),
    ("C311", "C312"),
    ("C312", "C313"),
    ("C313", "C314"),
    ("C314", "C315"),
    ("C315", "C316"),
    ("C316", "C317"),
    ("C317", "C318"),  # C318 is the end of the arm
    # Arm 2
    ("C2",   "O21"),
    ("O21",  "C21"),
    ("C21",  "O22"),  # Small O arm
    ("C21",  "C22"),
    ("C22",  "C23"),
    ("C23",  "C24"),
    ("C24",  "C25"),
    ("C25",  "C26"),
    ("C26",  "C27"),
    ("C27",  "C28"),
    ("C28",  "C29"),
    ("C29",  "C210"),
    ("C210", "C211"),
    ("C211", "C212"),
    ("C212", "C213"),
    ("C213", "C214"),
    ("C214", "C215"),
    ("C215", "C216"),
    ("C216", "C217"),
    ("C217", "C218"),  # C218 is the end of the arm
]


DOPC_CG_MAPPING = [
    ("NC3",   "PO4"),   # N is the start
    ("PO4",   "GL1"),
    ("GL1",   "GL2"),
    # First arm
    ("GL1",   "C1A"),
    ("C1A",   "D2A"),
    ("D2A",   "C3A"),
    ("C3A",   "C4A"),
    # Second arm
    ("GL2",   "C1B"),
    ("C1B",   "D2B"),
    ("D2B",   "C3B"),
    ("C3B",   "C4B"),
]
