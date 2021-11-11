# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 19:08:37 2021

@author: PENG Feng
@email:  im.pengf@outlook.com
"""

from my_utils import get_keybert_model, drop_dup_kws, write_result_to_html, sort_kws
from load_data import df_thesis
from tqdm import tqdm
import pickle
from yake import yake
from yake.highlight import TextHighlighter
import time


p_data = "./data"
f_thesis_keybert_p = "{}/thesis/keywords_keyBert_mpnet.pickle".format(p_data)
f_thesis_yake_p = "{}/thesis/keywords_yake3.pickle".format(p_data)

p_html = "./labeled_text"


TEST_BERT_YAKE = [False, False]

max_ngram_size = 3

# BERT settings
ONE_BY_ONE = True
top_n = 20 * max_ngram_size - 10
keyphrase_length = (1, max_ngram_size)
mmr = True
maxsum = not mmr

# YAKE settings


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


if TEST_BERT_YAKE[0]:
    model = get_keybert_model()

    if ONE_BY_ONE:
        l_thesis_kws = []
        for idx in tqdm(range(df_thesis.shape[0])):
            s_url, s_kws_truth, s_ab = df_thesis.iloc[idx, :]
            l_kws_truth = s_kws_truth.split("; ")

            l_kws_test = get_no_dup_kws(s_ab, model)

            n_kws_test = len(l_kws_test)
            n_kws_truth = len(l_kws_truth)
            if n_kws_test < n_kws_truth:
                print("!test word counts: {} < {}".format(n_kws_test, n_kws_truth))
            else:
                l_kws_test = l_kws_test[:n_kws_truth]
            l_thesis_kws.append(l_kws_test)
    else:
        l_ab = df_thesis.loc[:, "abstract"]
        l_thesis_kws = get_no_dup_kws_from_list(l_ab, model)
        for idx in tqdm(range(df_thesis.shape[0])):
            s_url, s_kws_truth, s_ab = df_thesis.iloc[idx, :]
            l_kws_truth = s_kws_truth.split("; ")
            l_kws_test = l_thesis_kws[idx]
            n_kws_test = len(l_kws_test)
            n_kws_truth = len(l_kws_truth)
            if n_kws_test < n_kws_truth:
                print("!test word counts: {} < {}".format(n_kws_test, n_kws_truth))
            else:
                l_kws_test = l_kws_test[:n_kws_truth]
            l_thesis_kws[idx] = l_kws_test

    df_thesis_keybert = df_thesis.iloc[: len(l_thesis_kws), :]
    df_thesis_keybert["KWG"] = l_thesis_kws

    df_thesis_keybert["KW"] = df_thesis_keybert["KW"].apply(lambda x: x.split("; "))

    with open(f_thesis_keybert_p, "wb") as f:
        pickle.dump(df_thesis_keybert, f)


if TEST_BERT_YAKE[1]:
    pyake = yake.KeywordExtractor(lan="en", n=3)

    l_thesis_kws = []
    for idx in tqdm(range(df_thesis.shape[0])):
        s_url, s_kws_truth, s_ab = df_thesis.iloc[idx, :]
        l_kws_truth = s_kws_truth.split("; ")

        l_kws_test = pyake.extract_keywords(s_ab)
        l_kws_test = sort_kws(l_kws_test)

        n_kws_test = len(l_kws_test)
        n_kws_truth = len(l_kws_truth)
        if n_kws_test < n_kws_truth:
            print("!test word counts: {} < {}".format(n_kws_test, n_kws_truth))
        else:
            l_kws_test = l_kws_test[:n_kws_truth]
        l_thesis_kws.append(l_kws_test)

    df_thesis_yake = df_thesis.iloc[: len(l_thesis_kws), :]
    df_thesis_yake["KWG"] = l_thesis_kws

    df_thesis_yake["KW"] = df_thesis_yake["KW"].apply(lambda x: x.split("; "))

    with open(f_thesis_yake_p, "wb") as f:
        pickle.dump(df_thesis_yake, f)


if not TEST_BERT_YAKE[0] and not TEST_BERT_YAKE[1]:
    # s_ab = """
    #     Supervised learning is the machine learning task of learning a function that
    #     maps an input to an output based on example input-output pairs.[1] It infers a
    #     function from labeled training data consisting of a set of training examples.[2]
    #     In supervised learning, each example is a pair consisting of an input object
    #     (typically a vector) and a desired output value (also called the supervisory signal).
    #     A supervised learning algorithm analyzes the training data and produces an inferred function,
    #     which can be used for mapping new examples. An optimal scenario will allow for the
    #     algorithm to correctly determine the class labels for unseen instances. This requires
    #     the learning algorithm to generalize from the training data to unseen situations in a
    #     'reasonable' way (see inductive bias).
    # """

    for idx in tqdm(range(df_thesis.shape[0])):
        s_url, s_kws_truth, s_ab = df_thesis.iloc[idx, :]

        pyake = yake.KeywordExtractor(lan="en", n=3)
        pyake_result = pyake.extract_keywords(s_ab)
        pyake_result = sort_kws(pyake_result)

        model = get_keybert_model()
        model_result = get_no_dup_kws(s_ab, model)

        max_kw_count = min(len(model_result), len(pyake_result))
        model_result = model_result[:max_kw_count]
        pyake_result = pyake_result[:max_kw_count]

        write_result_to_html(
            p_html, s_ab, [pyake_result, model_result], ["YAKE", "BERT"], max_ngram_size
        )
