# ---------------------------------------------------------
# WORD vs CHARACTER vs SUBWORD TOKENIZATION
# ---------------------------------------------------------


word = "unbelievable"


# ---------------------------------------------------------
# WORD-LEVEL
#
# Mechanism:
#
# tokens = [whole_word]
#
# If the word is not in the vocabulary:
#
# unknown word -> [UNK]
# ---------------------------------------------------------

word_vocab = {
    "the",
    "cat",
    "sat",
    "believable",
    "[UNK]"
}


def word_tokenize(word, vocab):

    if word in vocab:
        return [word]

    return ["[UNK]"]


word_tokens = word_tokenize(
    word,
    word_vocab
)


print("=" * 60)
print("WORD-LEVEL TOKENIZATION")
print("=" * 60)

print("Input:")
print(word)

print("\nTokens:")
print(word_tokens)

print("\nToken count:")
print(len(word_tokens))


# ---------------------------------------------------------
# CHARACTER-LEVEL
#
# Mechanism:
#
# word = c1 c2 c3 ... cn
#
# tokens = [c1, c2, c3, ..., cn]
# ---------------------------------------------------------

def character_tokenize(word):

    return list(word)


character_tokens = character_tokenize(word)


print("\n" + "=" * 60)
print("CHARACTER-LEVEL TOKENIZATION")
print("=" * 60)

print("Input:")
print(word)

print("\nTokens:")
print(character_tokens)

print("\nToken count:")
print(len(character_tokens))


# ---------------------------------------------------------
# TOY SUBWORD TOKENIZATION
#
# Mechanism:
#
# word -> reusable vocabulary pieces
#
# This is a BY-HAND demonstration.
#
# We are NOT learning the vocabulary yet.
# Later we will learn how algorithms such as BPE create
# these pieces automatically from a corpus.
# ---------------------------------------------------------

subword_vocab = {
    "un",
    "believ",
    "able"
}


def subword_tokenize_unbelievable(word):

    # By-hand split for this toy example
    if word == "unbelievable":
        return ["un", "believ", "able"]

    return list(word)


subword_tokens = subword_tokenize_unbelievable(
    word
)


print("\n" + "=" * 60)
print("SUBWORD TOKENIZATION")
print("=" * 60)

print("Input:")
print(word)

print("\nTokens:")
print(subword_tokens)

print("\nToken count:")
print(len(subword_tokens))


# ---------------------------------------------------------
# COMPARE TOKEN COUNTS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TOKEN COUNT COMPARISON")
print("=" * 60)

print(
    "Word-level:",
    len(word_tokens)
)

print(
    "Character-level:",
    len(character_tokens)
)

print(
    "Subword-level:",
    len(subword_tokens)
)


# ---------------------------------------------------------
# SHOW THE OOV PROBLEM
#
# Word tokenizer:
#
# unseen_word -> [UNK]
#
# The original information is lost.
# ---------------------------------------------------------

known_word = "cat"
unknown_word = "unbelievable"


print("\n" + "=" * 60)
print("WORD-LEVEL OOV CHECK")
print("=" * 60)

print(
    known_word,
    "->",
    word_tokenize(known_word, word_vocab)
)

print(
    unknown_word,
    "->",
    word_tokenize(unknown_word, word_vocab)
)


# ---------------------------------------------------------
# ATTENTION SIZE DEMO
#
# Approximate self-attention score matrix size:
#
# n x n
#
# Number of score entries:
#
# n^2
# ---------------------------------------------------------

short_sequence_length = 20
long_sequence_length = 80

short_attention_entries = (
    short_sequence_length ** 2
)

long_attention_entries = (
    long_sequence_length ** 2
)


print("\n" + "=" * 60)
print("SEQUENCE LENGTH / ATTENTION COST")
print("=" * 60)

print(
    "Short sequence tokens:",
    short_sequence_length
)

print(
    "Attention score entries:",
    short_attention_entries
)

print()

print(
    "Long sequence tokens:",
    long_sequence_length
)

print(
    "Attention score entries:",
    long_attention_entries
)


ratio = (
    long_attention_entries
    / short_attention_entries
)

print(
    "\nAttention matrix size ratio:",
    ratio
)