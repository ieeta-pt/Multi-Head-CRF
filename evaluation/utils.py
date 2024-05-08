#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os

import json


def get_filepaths(dir, ext):
    assert isinstance(dir, str)
    assert os.path.isdir(dir)
    assert ext in ('.ann', '.tsv', '.txt')
    #
    filenames = sorted(f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith(ext))
    #
    filepaths = [os.path.join(dir, f) for f in filenames]
    basenames = [f.rstrip(ext) for f in filenames]
    #
    return filepaths, filenames, basenames


def read_file(filepath):
    assert isinstance(filepath, str)
    fp = open(filepath, mode='r', encoding='utf-8')
    s = fp.read()
    fp.close()
    return s


def write_file(filepath, text):
    assert isinstance(filepath, str)
    assert isinstance(text, str)
    fp = open(filepath, mode='w', encoding='utf-8')
    _ = fp.write(text)
    fp.close()
