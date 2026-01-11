from enum import Enum


class JobType(str, Enum):
    word_count = "word_count"
    anagrams = "anagrams"
    line_statistics = "line_statistics"
    # TODO: ADD MONTE CARLO PI (PROBLEM: THE INPUT SIZE IS ALWAYS ONE-LINER, AND BREAKS THE ASSUMPTION OF LARGE TEXTS)
    # TODO: SUPPORT MORE JOB TYPES
