"""Stub for the real `fasttext` package (unbuildable on this machine — needs MSVC
headers that aren't present). OneRestore's utils_word_embedding.py imports
`fasttext.util` at module scope but our inference path only exercises the
`glove` word-embedding branch, never `load_fasttext_embeddings`. This stub
lets the import succeed; calling anything here is a deliberate hard failure.
"""


def load_model(*_args, **_kwargs):
    raise RuntimeError("fasttext is stubbed out — glove word-embedding path should be used instead")
