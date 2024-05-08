#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy

from preprocessing import assert_valid_span
from preprocessing import span_overlaps_span


ABC_LOWER = 'abcdefghijklmnopqrstuvwxyz'
ABC_UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
VALID = ABC_UPPER + ABC_LOWER + '_'


def is_valid_entity_type(typ):
    if not isinstance(typ, str) or (len(typ) < 1):
        return False
    for c in typ:
        if c not in VALID:
            return False
    return True


class Entity:
    #
    # This Entity class only allows to save entities with a single
    # (therefore, contiguous) text span.
    #
    def __init__(self, string, span, typ):
        assert isinstance(string, str)
        assert_valid_span(span)
        begin, end = span
        n = end - begin
        assert len(string) == n
        assert is_valid_entity_type(typ)
        #
        self.string = string
        self.span = span
        self.begin = begin
        self.end = end
        self.n_characters = n
        self.typ = typ
    #
    def __repr__(self):
        return 'Entity{}'.format((self.string, self.span, self.typ))
    #
    def __len__(self):
        return self.n_characters
    #
    def __eq__(self, other):
        #
        # Compare two entities to check if they have (i) the same
        # string, (ii) the same span, and (iii) the same type.
        #
        assert isinstance(other, Entity)
        if ((self.string == other.string) and
            (self.span == other.span) and (self.typ == other.typ)):
            return True
        else:
            return False
    #
    def __hash__(self):
        return hash((self.string, self.span, self.typ))
    #
    def __lt__(self, other):
        #
        # This magic method allows to easily sort a list of Entity
        # objects with the sorted() function.
        #
        # Attributes sorting priority:
        #   1. begin
        #   2. end
        #   3. typ
        #   4. string
        #
        assert isinstance(other, Entity)
        if self.begin < other.begin:
            return True
        elif self.begin == other.begin:
            if self.end < other.end:
                return True
            elif self.end == other.end:
                if self.typ < other.typ:
                    return True
                elif self.typ == other.typ:
                    if self.string < other.string:
                        return True
        return False
    #
    def coextensive(self, other):
        #
        # Two annotations are coextensive if they hit the same span,
        # that is, their start and end oﬀsets are equal. Reference:
        # https://gate.ac.uk/sale/tao/splitch10.html
        #
        assert isinstance(other, Entity)
        if self.span == other.span:
            return True
        else:
            return False
    #
    def overlaps(self, other):
        assert isinstance(other, Entity)
        if span_overlaps_span(self.span, other.span):
            return True
        else:
            return False


class EntitySet:
    #
    # A set of entities. Overlapping of entities is allowed.
    # The overlapping problem is expected to be solved afterwards, when
    # converting the corpus to the BIO format.
    #
    def __init__(self):
        self.entities = set()
    #
    def __len__(self):
        return len(self.entities)
    #
    def __str__(self):
        s = ''
        for e in self.get():
            s += '{}\n'.format(e)
        return s.strip()
    #
    def __eq__(self, other):
        assert isinstance(other, EntitySet)
        return self.entities == other.entities
    #
    def __iter__(self):
        for e in self.entities:
            yield e
    #
    def has(self, e):
        assert isinstance(e, Entity)
        return e in self.entities
    #
    def add(self, e):
        assert isinstance(e, Entity)
        self.entities.add(e)
    #
    def get(self, typ=None):
        assert (typ is None) or is_valid_entity_type(typ)
        #
        # If type is not None, get the entities of the specified type.
        #
        if typ is None:
            entities = self.entities
        else:
            entities = [e for e in self.entities if e.typ == typ]
        #
        # Sort entities.
        #
        entities = sorted(entities)
        #
        # Return a deepcopy.
        #
        return deepcopy(entities)
    #
    def types(self):
        return sorted(set(e.typ for e in self.entities))

    def get_overlapping_entities(self):
        n = len(self.entities)
        sorted_entities = sorted(self.entities)
        #
        pairs_with_overlapping_entities = list()
        overlapping_entities = EntitySet()
        #
        for i in range(n-1):
            for j in range(i+1, n):
                e1 = sorted_entities[i]
                e2 = sorted_entities[j]
                if e1.overlaps(e2):
                    pairs_with_overlapping_entities.append((e1, e2))
                    overlapping_entities.add(e1)
                    overlapping_entities.add(e2)
        #
        return deepcopy(pairs_with_overlapping_entities), overlapping_entities.get()
