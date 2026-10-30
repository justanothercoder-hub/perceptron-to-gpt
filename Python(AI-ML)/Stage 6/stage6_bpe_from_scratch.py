import re
from collections import Counter

# BYTE-PAIR ENCODING (BPE) FROM SCRATCH
# Topic 2.1 — Character-level BPE
# ---------------------------------------------------------


# ---------------------------------------------------------
# TRAINING CORPUS
#
# word -> frequency
# ---------------------------------------------------------

corpus = {
    "low": 5,
    "lower": 2,
    "newest": 6,
    "widest": 3,
}


# ---------------------------------------------------------
# Mechanism:
#
# word -> characters + end-of-word marker
#
# Example:
#
# "low"
# ->
# ('l', 'o', 'w', '</w>')
# ---------------------------------------------------------

def make_training_vocab(corpus):

    training_vocab = {}

    for word, frequency in corpus.items():

        symbols = tuple(
            list(word) + ["</w>"]
        )

        training_vocab[symbols] = frequency

    return training_vocab


# ---------------------------------------------------------
# Formula / mechanism:
#
# count(a, b) =
#     sum of frequencies of every word representation
#     containing adjacent pair (a, b)
#
# Example:
#
# newest x6
#
# ('e', 's') contributes +6
# ---------------------------------------------------------

def get_pair_counts(training_vocab):

    pair_counts = {}

    for symbols, frequency in training_vocab.items():

        for i in range(len(symbols) - 1):

            pair = (
                symbols[i],
                symbols[i + 1]
            )

            if pair not in pair_counts:
                pair_counts[pair] = 0

            pair_counts[pair] += frequency

    return pair_counts


# ---------------------------------------------------------
# Mechanism:
#
# For pair:
#
# (A, B)
#
# replace every adjacent:
#
# A B
#
# with:
#
# AB
#
# Example:
#
# ('e', 's')
#
# n e w e s t </w>
#
# becomes:
#
# n e w es t </w>
# ---------------------------------------------------------

def merge_symbols(symbols, pair):

    first, second = pair

    merged = []

    i = 0

    while i < len(symbols):

        if (
            i < len(symbols) - 1
            and symbols[i] == first
            and symbols[i + 1] == second
        ):

            merged.append(
                first + second
            )

            i += 2

        else:

            merged.append(
                symbols[i]
            )

            i += 1

    return tuple(merged)


# ---------------------------------------------------------
# Mechanism:
#
# Apply one learned merge rule to EVERY word
# representation in the training corpus.
#
# A B -> AB
# ---------------------------------------------------------

def merge_pair(pair, training_vocab):

    new_vocab = {}

    for symbols, frequency in training_vocab.items():

        new_symbols = merge_symbols(
            symbols,
            pair
        )

        if new_symbols not in new_vocab:
            new_vocab[new_symbols] = 0

        new_vocab[new_symbols] += frequency

    return new_vocab


# ---------------------------------------------------------
# BPE TRAINING MECHANISM:
#
# Repeat K times:
#
# 1. Count every adjacent pair
#
# 2. Find the pair with maximum weighted frequency
#
#       best_pair = argmax count(pair)
#
# 3. Record that pair as a merge rule
#
# 4. Replace:
#
#       A B -> AB
#
#    everywhere
#
# 5. Recount pairs from the NEW representation
# ---------------------------------------------------------

def train_bpe(corpus, num_merges):

    training_vocab = make_training_vocab(
        corpus
    )

    merge_rules = []

    print("=" * 70)
    print("INITIAL TRAINING VOCAB")
    print("=" * 70)

    for symbols, frequency in training_vocab.items():

        print(
            symbols,
            "x",
            frequency
        )

    for step in range(num_merges):

        pair_counts = get_pair_counts(
            training_vocab
        )

        if len(pair_counts) == 0:
            break

        # If two pairs have the same count,
        # Python keeps the first encountered maximum.
        best_pair = max(
            pair_counts,
            key=pair_counts.get
        )

        best_count = pair_counts[
            best_pair
        ]

        merge_rules.append(
            best_pair
        )

        print("\n" + "=" * 70)
        print(f"MERGE {step + 1}")
        print("=" * 70)

        print(
            "Best pair:",
            best_pair
        )

        print(
            "Count:",
            best_count
        )

        training_vocab = merge_pair(
            best_pair,
            training_vocab
        )

        print("\nTraining vocabulary after merge:")

        for symbols, frequency in training_vocab.items():

            print(
                symbols,
                "x",
                frequency
            )

    return (
        merge_rules,
        training_vocab
    )


# ---------------------------------------------------------
# ENCODING MECHANISM:
#
# Encoding DOES NOT learn new merges.
#
# 1. Start from:
#
#       characters + </w>
#
# 2. Replay the learned merge rules
#    IN THE SAME ORDER they were learned.
#
# 3. Stop after all learned rules have been tried.
#
# Final remaining symbols = BPE tokens
# ---------------------------------------------------------

def encode_word(word, merge_rules):

    symbols = tuple(
        list(word) + ["</w>"]
    )

    print("\nStarting symbols:")

    print(symbols)

    for step, pair in enumerate(
        merge_rules,
        start=1
    ):

        old_symbols = symbols

        symbols = merge_symbols(
            symbols,
            pair
        )

        if symbols != old_symbols:

            print(
                f"After merge {step} {pair}:"
            )

            print(symbols)

    return list(symbols)


# =========================================================
# INITIAL BASE VOCABULARY
# =========================================================

base_symbols = {
    "</w>"
}

for word in corpus:

    for character in word:
        base_symbols.add(character)


print("=" * 70)
print("INITIAL BASE SYMBOL VOCABULARY")
print("=" * 70)

print(
    sorted(base_symbols)
)

print(
    "\nBase vocabulary size:",
    len(base_symbols)
)


# =========================================================
# TRAIN BPE
# =========================================================

NUM_MERGES = 8

merge_rules, final_training_vocab = train_bpe(
    corpus,
    NUM_MERGES
)


# =========================================================
# PRINT LEARNED MERGE RULES
# =========================================================

print("\n" + "=" * 70)
print("LEARNED MERGE RULES")
print("=" * 70)

for number, pair in enumerate(
    merge_rules,
    start=1
):

    print(
        f"{number}. {pair}"
    )


# =========================================================
# FINAL TRAINING REPRESENTATIONS
# =========================================================

print("\n" + "=" * 70)
print("FINAL TRAINING VOCABULARY")
print("=" * 70)

for symbols, frequency in final_training_vocab.items():

    print(
        symbols,
        "x",
        frequency
    )


# =========================================================
# ENCODE A WORD NEVER SEEN DURING TRAINING
# =========================================================

new_word = "lowest"


print("\n" + "=" * 70)
print("ENCODING UNSEEN WORD")
print("=" * 70)

print(
    "Word:",
    new_word
)


encoded_word = encode_word(
    new_word,
    merge_rules
)


print(
    "\nFinal BPE tokens:"
)

print(
    encoded_word
)


# =========================================================
# VERIFICATION
# =========================================================

expected_first_three_merges = [
    ("e", "s"),
    ("es", "t"),
    ("est", "</w>")
]


expected_lowest_tokens = [
    "low",
    "est</w>"
]


print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)


print(
    "First 3 merges correct:",
    merge_rules[:3]
    == expected_first_three_merges
)


print(
    "Unseen word encoding correct:",
    encoded_word
    == expected_lowest_tokens
)



# =========================================================
# TOPIC 2.2 — BYTE-LEVEL BPE
# =========================================================


# ---------------------------------------------------------
# PRE-TOKENIZATION
#
# Mechanism:
#
# Full text -> smaller word-like chunks
#
# Example:
#
# "how are you"
#
# ->
#
# ["how", " are", " you"]
#
# BPE will operate INSIDE each chunk only.
# ---------------------------------------------------------

def pretokenize(text):

    chunks = re.findall(
        r" ?\S+|\s+",
        text
    )

    return chunks


# =========================================================
# TEST PRE-TOKENIZATION
# =========================================================

sample_text = "how are you"


print("\n" + "=" * 70)
print("PRE-TOKENIZATION")
print("=" * 70)

print(
    "Original text:",
    repr(sample_text)
)

print(
    "Chunks:",
    pretokenize(sample_text)
)


# =========================================================
# UTF-8 BYTE DEMONSTRATION
# =========================================================


# ---------------------------------------------------------
# Mechanism:
#
# UTF-8 encoding:
#
# text character
# ->
# one or more bytes
#
# Each byte is an integer:
#
# 0 <= byte <= 255
#
# Therefore there are exactly:
#
# 2^8 = 256
#
# possible base byte values.
# ---------------------------------------------------------

def show_utf8_bytes(text):

    raw_bytes = text.encode(
        "utf-8"
    )

    byte_values = list(
        raw_bytes
    )

    print(
        repr(text),
        "->",
        byte_values
    )

    return byte_values


print("\n" + "=" * 70)
print("UTF-8 BYTE EXAMPLES")
print("=" * 70)


show_utf8_bytes("A")

show_utf8_bytes("न")

show_utf8_bytes("🙂")

# =========================================================
# BYTE-LEVEL TRAINING VOCABULARY
# =========================================================


# ---------------------------------------------------------
# Mechanism:
#
# 1. Pre-tokenize text into chunks
#
# 2. Convert each chunk into UTF-8 bytes
#
# 3. Split those bytes into individual byte symbols
#
# Example:
#
# "cat"
#
# UTF-8:
#
# [99, 97, 116]
#
# Initial byte symbols:
#
# (b'c', b'a', b't')
# ---------------------------------------------------------

def make_byte_training_vocab(texts):

    chunk_counts = Counter()

    for text in texts:

        chunks = pretokenize(
            text
        )

        for chunk in chunks:

            chunk_counts[
                chunk
            ] += 1

    training_vocab = {}

    for chunk, frequency in chunk_counts.items():

        raw_bytes = chunk.encode(
            "utf-8"
        )

        symbols = tuple(
            bytes([byte_value])
            for byte_value in raw_bytes
        )

        training_vocab[
            symbols
        ] = frequency

    return training_vocab


# ---------------------------------------------------------
# Formula:
#
# count(A, B)
# =
# sum of frequencies of every chunk containing
# adjacent byte-token pair (A, B)
#
# Same mechanism as character BPE.
# ---------------------------------------------------------

def get_byte_pair_counts(training_vocab):

    pair_counts = {}

    for symbols, frequency in training_vocab.items():

        for i in range(
            len(symbols) - 1
        ):

            pair = (
                symbols[i],
                symbols[i + 1]
            )

            if pair not in pair_counts:

                pair_counts[
                    pair
                ] = 0

            pair_counts[
                pair
            ] += frequency

    return pair_counts


# ---------------------------------------------------------
# Mechanism:
#
# Merge adjacent byte tokens:
#
# A + B -> AB
#
# Example:
#
# b'c' + b'a'
# ->
# b'ca'
# ---------------------------------------------------------

def merge_byte_symbols(symbols, pair):

    first, second = pair

    merged = []

    i = 0

    while i < len(symbols):

        if (
            i < len(symbols) - 1
            and symbols[i] == first
            and symbols[i + 1] == second
        ):

            merged.append(
                first + second
            )

            i += 2

        else:

            merged.append(
                symbols[i]
            )

            i += 1

    return tuple(merged)



# ---------------------------------------------------------
# Mechanism:
#
# Apply one byte-pair merge to every chunk
# in the training vocabulary.
#
# A B -> AB
# ---------------------------------------------------------

def merge_byte_pair(pair, training_vocab):

    new_vocab = {}

    for symbols, frequency in training_vocab.items():

        new_symbols = merge_byte_symbols(
            symbols,
            pair
        )

        if new_symbols not in new_vocab:
            new_vocab[new_symbols] = 0

        new_vocab[new_symbols] += frequency

    return new_vocab


# =========================================================
# TRAIN BYTE-LEVEL BPE
# =========================================================


# ---------------------------------------------------------
# Mechanism:
#
# Repeat K times:
#
# 1. Count all adjacent byte-token pairs
#
# 2. Find the most frequent pair
#
#       best_pair = argmax count(pair)
#
# 3. Save that pair as a merge rule
#
# 4. Merge that pair everywhere
#
# 5. Repeat
#
# Same BPE algorithm as Topic 2.1.
# Only the starting symbols are bytes instead of characters.
# ---------------------------------------------------------

def train_byte_bpe(texts, num_merges):

    training_vocab = make_byte_training_vocab(
        texts
    )

    merge_rules = []

    for step in range(num_merges):

        pair_counts = get_byte_pair_counts(
            training_vocab
        )

        if len(pair_counts) == 0:
            break

        best_pair = max(
            pair_counts,
            key=pair_counts.get
        )

        best_count = pair_counts[
            best_pair
        ]

        merge_rules.append(
            best_pair
        )

        training_vocab = merge_byte_pair(
            best_pair,
            training_vocab
        )

        print(
            f"Merge {step + 1}:",
            best_pair,
            "count =",
            best_count
        )

    return merge_rules

# =========================================================
# BYTE-LEVEL BPE ENCODING
# =========================================================


# ---------------------------------------------------------
# Mechanism:
#
# text
# ->
# pre-tokenized chunks
# ->
# UTF-8 bytes
# ->
# individual byte symbols
# ->
# replay learned merge rules in order
#
# Encoding learns NO new rules.
# ---------------------------------------------------------

def encode_byte_bpe(text, merge_rules):

    final_tokens = []

    chunks = pretokenize(
        text
    )

    for chunk in chunks:

        raw_bytes = chunk.encode(
            "utf-8"
        )

        symbols = tuple(
            bytes([byte_value])
            for byte_value in raw_bytes
        )

        for pair in merge_rules:

            symbols = merge_byte_symbols(
                symbols,
                pair
            )

        final_tokens.extend(
            symbols
        )

    return final_tokens

# =========================================================
# BYTE-LEVEL BPE DECODING
# =========================================================


# ---------------------------------------------------------
# Mechanism:
#
# token_1 + token_2 + ... + token_n
#
# ->
# original UTF-8 byte sequence
#
# ->
# UTF-8 decode
#
# Therefore:
#
# decode(encode(text)) == text
# ---------------------------------------------------------

def decode_byte_bpe(tokens):

    raw_bytes = b"".join(
        tokens
    )

    return raw_bytes.decode(
        "utf-8"
    )

# =========================================================
# BYTE-BPE TRAINING TEXT
# =========================================================

byte_training_texts = [

    "hello world hello world",

    "how low can you go how low can you go",

    "नमस्ते दुनिया नमस्ते दुनिया",

    "hello 🙂 hello 🙂 🙂🚀"
]

BYTE_NUM_MERGES = 40


print("\n" + "=" * 70)
print("TRAINING BYTE-LEVEL BPE")
print("=" * 70)


byte_merge_rules = train_byte_bpe(
    byte_training_texts,
    BYTE_NUM_MERGES
)


print(
    "\nNumber of learned merges:",
    len(byte_merge_rules)
)

# =========================================================
# ROUND-TRIP VERIFICATION
# =========================================================


test_texts = {

    "English":
        "how low can you go?",

    "Hindi":
        "नमस्ते दुनिया",

    "Emoji":
        "Hello 🙂🚀"
}


print("\n" + "=" * 70)
print("ROUND-TRIP TESTS")
print("=" * 70)


all_passed = True


for label, text in test_texts.items():

    encoded = encode_byte_bpe(
        text,
        byte_merge_rules
    )

    decoded = decode_byte_bpe(
        encoded
    )

    passed = (
        decoded == text
    )

    if not passed:
        all_passed = False

    print(
        "\n",
        label
    )

    print(
        "Original:",
        text
    )

    print(
        "Character count:",
        len(text)
    )

    print(
        "UTF-8 byte count:",
        len(
            text.encode("utf-8")
        )
    )

    print(
        "BPE token count:",
        len(encoded)
    )

    print(
        "Encoded tokens:",
        encoded
    )

    print(
        "Decoded:",
        decoded
    )

    print(
        "Round-trip correct:",
        passed
    )


print("\n" + "=" * 70)
print("FINAL CHECK")
print("=" * 70)


print(
    "English + Hindi + Emoji all exact:",
    all_passed
)
