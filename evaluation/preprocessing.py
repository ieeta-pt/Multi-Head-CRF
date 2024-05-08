#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def assert_valid_span(span):
    assert isinstance(span, tuple)
    assert len(span) == 2
    begin, end = span
    assert isinstance(begin, int) and (begin >= 0)
    assert isinstance(end, int) and (end > begin)


def assert_valid_spans(spans):
    assert isinstance(spans, list)
    for s in spans:
        assert_valid_span(s)


def span_overlaps_span(span1, span2):
    assert_valid_span(span1)
    assert_valid_span(span2)
    if (span1[1] <= span2[0]) or (span1[0] >= span2[1]):
        return False
    return True


def span_contains_span(span1, span2):
    assert_valid_span(span1)
    assert_valid_span(span2)
    if (span1[0] <= span2[0]) and (span1[1] >= span2[1]):
        return True
    return False


def spans_are_disjoint(spans):
    assert_valid_spans(spans)
    n = len(spans)
    for i in range(n):
        for j in range(i + 1, n):
            if span_overlaps_span(spans[i], spans[j]):
                return False
    return True


def spans_are_disjoint_and_sorted(spans):
    assert_valid_spans(spans)
    if len(spans) > 0:
        previous_end = spans[0][1]
        for (begin, end) in spans[1:]:
            if begin < previous_end:
                return False
            previous_end = end
    return True


def sort_spans(spans, *args):
    assert_valid_spans(spans)
    n = len(spans)
    for a in args:
        assert len(a) == n
    #
    z = zip(spans, *args)
    z = sorted(z, key=lambda z: z[0][1])
    spans, *args = zip(*sorted(z, key=lambda z: z[0][0]))
    #
    spans = list(spans)
    for i in range(len(args)):
        args[i] = list(args[i])
    #
    return spans, *args
