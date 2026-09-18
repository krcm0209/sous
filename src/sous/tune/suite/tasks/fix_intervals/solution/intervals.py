"""merge(intervals) merges a list of closed integer intervals [start, end]:
the result is sorted by start, and any two intervals that overlap or touch
(one's start equal to the other's end) become one interval."""


def merge(intervals: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
