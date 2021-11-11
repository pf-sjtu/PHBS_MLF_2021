# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 19:52:58 2021

@author: PENG Feng
@email:  im.pengf@outlook.com
"""

import pickle

f_thesis_keybert_p = "./keywords_keyBert.pickle"
f_thesis_keybert3_p = "./keywords_keyBert3.pickle"

with open(f_thesis_keybert_p, "rb") as f:
    df_thesis_keybert = pickle.load(f)

with open(f_thesis_keybert3_p, "rb") as f:
    df_thesis_keybert3 = pickle.load(f)
