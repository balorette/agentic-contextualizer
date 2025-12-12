"""Discovery phase for scoped context generation."""

import re
from typing import List, Set

# Common stopwords to filter out
STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
    "because", "until", "while", "this", "that", "these", "those", "what",
    "which", "who", "whom", "functionality", "feature", "features", "work",
    "works", "use", "uses", "using",
}

MIN_KEYWORD_LENGTH = 3


def extract_keywords(question: str) -> List[str]:
    """Extract meaningful keywords from a scope question.

    Args:
        question: The user's scope question

    Returns:
        List of lowercase keywords, filtered and deduplicated
    """
    # Normalize: lowercase and split on non-alphanumeric
    words = re.split(r"[^a-zA-Z0-9]+", question.lower())

    # Filter
    keywords = []
    seen: Set[str] = set()

    for word in words:
        if (
            word
            and len(word) >= MIN_KEYWORD_LENGTH
            and word not in STOPWORDS
            and word not in seen
        ):
            keywords.append(word)
            seen.add(word)

    return keywords
