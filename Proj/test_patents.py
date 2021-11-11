# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 19:08:37 2021

@author: PENG Feng
@email:  im.pengf@outlook.com
"""

from my_utils import get_keybert_model, drop_dup_kws
from load_data import df_patent
from tqdm import tqdm
import pickle


p_data = "./data"
f_patent_keybert_p = "{}/patents/keywords_keyBert_mpnet.pickle".format(p_data)


# BERT settings
ONE_BY_ONE = True
keyphrase_length = (1, 2)
top_n = 30
mmr = True
maxsum = not mmr


def get_no_dup_kws(s, keybert_model):
    global keyphrase_length, mmr, maxsum, top_n

    keywords = keybert_model._extract_keywords_single_doc(
        s,
        top_n=top_n,
        keyphrase_ngram_range=keyphrase_length,
        use_mmr=mmr,
        use_maxsum=maxsum,
        diversity=0.5,
        stop_words=None,
    )
    no_dup_keywords = drop_dup_kws(keywords, keybert_model.model, thd=0.6)
    return no_dup_keywords


def get_no_dup_kws_from_list(l_s, keybert_model):
    global keyphrase_length, top_n

    keywords = keybert_model._extract_keywords_multiple_docs(
        l_s, top_n=top_n, keyphrase_ngram_range=keyphrase_length, stop_words=None
    )
    no_dup_keywords = [
        drop_dup_kws(i, keybert_model.model, thd=0.6) for i in tqdm(keywords)
    ]
    return no_dup_keywords


model = get_keybert_model()

if ONE_BY_ONE:
    l_patent_kws = []
    for idx in tqdm(df_patent.index):
        s_ab = df_patent.loc[idx, "abstract"]
        l_kws_test = get_no_dup_kws(s_ab, model)
        l_patent_kws.append(l_kws_test)
else:
    l_ab = df_patent.loc[:, "abstract"]
    l_patent_kws = get_no_dup_kws_from_list(l_ab, model)

df_patent_keybert = df_patent.iloc[: len(l_patent_kws), :]
df_patent_keybert["KWG"] = l_patent_kws

with open(f_patent_keybert_p, "wb") as f:
    pickle.dump(df_patent_keybert, f)
