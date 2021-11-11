# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 19:09:43 2021

@author: PENG Feng
@email:  im.pengf@outlook.com
"""
from keybert import KeyBERT
from tests.utils import get_test_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
import re
import torch
from yake.highlight import TextHighlighter
from adjustText import adjust_text
from astroML.plotting import scatter_contour
from matplotlib import pyplot as plt
from matplotlib import ticker
from tqdm import tqdm
import math


# nltk.set_proxy("http://127.0.0.1:10809")
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

s_test = (
    "to for at about on since because correctly some supervised learning however 100"
)
tokens = nltk.word_tokenize(s_test)
pos_tags = nltk.pos_tag(tokens)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = {
    "keyBert": "distilbert-base-nli-mean-tokens",
    "keyBert_mpnet": "paraphrase-mpnet-base-v2",
}


def noun(s):
    re_not_noun_beg = r"IN"
    re_not_noun_end = r"IN|JJ.*|RB.*|VBD"
    re_not_noun_all = r"CC|EX|SYM|UH|MD|TO|W.*|DT|PRP.*|PDT|DT|RP|VBP|VBZ|CD"
    re_not_noun_beg = "|".join([re_not_noun_beg, re_not_noun_all])
    re_not_noun_end = "|".join([re_not_noun_end, re_not_noun_all])
    tokens = nltk.word_tokenize(s)
    pos_tag_beg = nltk.pos_tag([tokens[0]])
    pos_tag_end = nltk.pos_tag([tokens[-1]])
    # pos_tags = nltk.pos_tag(tokens)
    attr_beg = pos_tag_beg[0][-1]
    attr_end = pos_tag_end[0][-1]
    if re.match(re_not_noun_beg, attr_beg) is not None:
        return False
    if re.match(re_not_noun_end, attr_end) is not None:
        return False
    return True


def sort_kws(kws):
    a_kws = np.array(kws)
    sorted_kws = [kws[idx] for idx in np.argsort(a_kws[:, 1])]
    sorted_kws.reverse()
    return sorted_kws


def word_vec(s, model):
    embed = model.embed(s)
    if len(embed.shape) == 2:
        embed = embed.mean(0)
    embed = embed[None, :]
    return embed


def word_cosine_similarity(s1, s2, model):
    embeds = []
    for s in [s1, s2]:
        embeds.append(word_vec(s, model))
    similarity = cosine_similarity(embeds[0], embeds[1])[0][0]
    # print("SIM: {} & {} = {:.3f}".format(s1, s2, similarity))
    return similarity


def norm_importance_to_length(l_kws):
    func = lambda x: (x[0], round(x[1] / (len(x[0].split(" ")) + 4) * 5, 5))
    return [func(i) for i in l_kws]


def drop_dup_kws(kws, model, thd=0.5):
    if len(kws) > 1:
        sorted_kws = sort_kws(kws)
        l_kws = [i for i in sorted_kws if noun(i[0])]
        l_kws_vec = [word_vec(i, model) for i in l_kws]
        dropped_idx = set()
        for i in range(len(l_kws)):
            if i not in dropped_idx:
                kw_vec1 = l_kws_vec[i]
                for j in range(i + 1, len(l_kws)):
                    if j not in dropped_idx:
                        kw_vec2 = l_kws_vec[j]
                        similarity = cosine_similarity(kw_vec1, kw_vec2)[0][0]
                        if similarity > thd:
                            dropped_idx.add(j)
                            # print("BAN")
        dropped_kws = [l_kws[i] for i in range(len(l_kws)) if i not in dropped_idx]
        return dropped_kws
    elif len(kws) == 1 and not isinstance(kws[0], tuple):
        kws = []
    return kws


def get_keybert_model(model="paraphrase-mpnet-base-v2"):
    keybert_model = KeyBERT(model=model)
    keybert_model.model.embedding_model.to(device)
    return keybert_model


def highlight(text, keywords, max_ngram_size):
    if isinstance(keywords, list) and len(keywords) > 0:
        if isinstance(keywords[0], tuple):
            keywords = [kw[0] for kw in keywords]
        th = TextHighlighter(max_ngram_size=max_ngram_size)
        textHighlighted = th.highlight(text, keywords)
        return textHighlighted
    return None


def write_result_to_html(p_html, text, keywords, titles, max_ngram_size):
    if not isinstance(keywords, list):
        keywords = [keywords]
    if not isinstance(titles, list):
        titles = [titles]
    if not isinstance(text, list):
        text = [text] * len(keywords)
    assert len(text) == len(keywords)
    assert len(text) == len(titles)

    l_html = []
    for tx, kws, title in zip(text, keywords, titles):
        s_hl = highlight(tx, kws, max_ngram_size)
        l_html.append("<method>{}</method>{}".format(title, s_hl))

    with open("{}/template.html".format(p_html), "r", encoding="utf-8") as f:
        s_tempalte = f.read()

    s_html = s_tempalte + "\n".join(l_html)

    with open("{}/index.html".format(p_html), "w", encoding="utf-8") as f:
        f.write(s_html)


def get_sim_dots(df_kws, sim_dots_imp_int=0.5):
    l_sim_dots = []
    df_kws2 = df_kws.dropna()
    for idx in tqdm(df_kws2.index):
        x1, x2, imp = df_kws2.loc[idx, ["X1", "X2", "importance"]]
        n_dots = math.floor(imp / sim_dots_imp_int)
        for _ in range(n_dots):
            l_sim_dots += [(x1, x2)] * n_dots
    a_sim_dots = np.array(l_sim_dots)
    return a_sim_dots


def plot_kws(
    fig,
    ax,
    color_func,
    df_plot_kws=None,
    bins=30,
    a_sim_dots=None,
    n_comm_kws=5,
    l_clust=None,
    l_df_period=None,
    year_int=5,
    period_arrow=False,
    period_line=True,
    kw_color_func_flag=False,
    p_data="./data",
    file_name="clust_map",
):
    scatter = df_plot_kws is not None
    arrow_color = "darkgrey" if not scatter else "black"
    arrow_label_color = "white" if not scatter else "black"

    if scatter:
        ax.scatter(
            df_plot_kws["X1"],
            df_plot_kws["X2"],
            c=df_plot_kws["clust"].apply(color_func),
            s=df_plot_kws["importance"],
            alpha=0.5,
        )

    else:
        ax.set_facecolor("midnightblue")
        l_x, l_y = a_sim_dots.T

        pset, cset = scatter_contour(
            l_x,
            l_y,
            threshold=200,
            log_counts=True,
            ax=ax,
            histogram2d_args=dict(bins=bins),
            plot_args=dict(marker=",", linestyle="none", color="slateblue"),
            contour_args=dict(cmap=plt.cm.plasma),
        )
        fig.colorbar(cset, ax=ax)

    annotate_text = []
    if l_clust is not None:
        for clust in l_clust:
            # pos_center = clust["center"]
            df_comm_kw = clust["df_kws_clust"].iloc[:n_comm_kws, :]
            for i_word in df_comm_kw.index:
                name = df_comm_kw.loc[i_word, "KW"]
                x = df_comm_kw.loc[i_word, "X1"]
                y = df_comm_kw.loc[i_word, "X2"]
                annotate_text.append(
                    ax.annotate(
                        name,
                        (x, y),
                        color=color_func(clust["i"])
                        if kw_color_func_flag
                        else arrow_label_color,
                    )
                )

    if l_df_period is not None:
        l_pos_weight = []
        for i in range(len(l_df_period)):
            df_p_kws = l_df_period[i]["df_kws"]
            df_comm_kw = df_p_kws.sort_values("importance", ascending=False)
            df_comm_kw = df_comm_kw.iloc[:n_comm_kws, :]
            s_comm_kw = ", ".join(df_comm_kw["KW"])
            l_df_period[i]["df_comm_kw"] = df_comm_kw
            l_df_period[i]["s_comm_kw"] = s_comm_kw
            l_pos_weight.append(l_df_period[i]["pos_weight"])

        a_pos_weight = np.array(l_pos_weight)

        if period_line:
            ax.plot(a_pos_weight[:, 0], a_pos_weight[:, 1], lw=4, color=arrow_color)

        for i in range(len(l_df_period)):
            n_dt_beg = l_df_period[i]["dt_beg"].year
            x = a_pos_weight[i, 0]
            y = a_pos_weight[i, 1]
            if i < len(l_df_period) - 1:
                if period_arrow:
                    dx = a_pos_weight[i + 1, 0] - x
                    dy = a_pos_weight[i + 1, 1] - y
                    ax.arrow(
                        x,
                        y,
                        dx,
                        dy,
                        width=0.05,
                        head_width=0.08,
                        ec=arrow_color,
                        fc=arrow_color,
                        length_includes_head=True,
                        shape="left",
                    )
                s_year = "{}-{}".format(n_dt_beg, n_dt_beg + year_int)
            else:
                s_year = "{}-now".format(n_dt_beg)

            annotate_text.append(
                ax.annotate(s_year, (x, y), color=arrow_label_color, fontweight="bold")
            )

    adjust_text(
        annotate_text,
        arrowprops=dict(
            arrowstyle="-",
            lw=1,
            color=arrow_label_color,
        ),
    )

    ax.set_xticks([])
    ax.set_yticks([])

    if file_name is not None:
        # fig.savefig("{}/plot/{}.pdf".format(p_data, file_name), bbox_inches="tight")
        fig.savefig(
            "{}/plot/{}.png".format(p_data, file_name), bbox_inches="tight", dpi=200
        )
        fig.show()


if __name__ == "__main__":
    keybert_model = get_keybert_model("paraphrase-mpnet-base-v2")

    doc_one = """
              Supervised learning is the machine learning task of learning a function that
              maps an input to an output based on example input-output pairs.[1] It infers a
              function from labeled training data consisting of a set of training examples.[2]
              
          """
    # doc_one = """
    #           Supervised learning is the machine learning task of learning a function that
    #           maps an input to an output based on example input-output pairs.[1] It infers a
    #           function from labeled training data consisting of a set of training examples.[2]
    #           In supervised learning, each example is a pair consisting of an input object
    #           (typically a vector) and a desired output value (also called the supervisory signal).
    #           A supervised learning algorithm analyzes the training data and produces an inferred function,
    #           which can be used for mapping new examples. An optimal scenario will allow for the
    #           algorithm to correctly determine the class labels for unseen instances. This requires
    #           the learning algorithm to generalize from the training data to unseen situations in a
    #           'reasonable' way (see inductive bias).
    #       """

    # doc_one, doc_two = get_test_data()

    keyphrase_length = (1, 3)  # range(5)
    vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words="english")
    mmr = True
    maxsum = not mmr
    top_n = 50

    # keywords = base_keybert.extract_keywords(doc_one, keyphrase_ngram_range=(1, 2), stop_words=None)

    keywords = keybert_model._extract_keywords_single_doc(
        doc_one,
        top_n=top_n,
        keyphrase_ngram_range=keyphrase_length,
        use_mmr=mmr,
        use_maxsum=maxsum,
        diversity=0.5,
        stop_words=None,
    )

    keywords = norm_importance_to_length(keywords)

    print("A:\n", sort_kws(keywords))
    print("B:\n", drop_dup_kws(keywords, keybert_model.model, 0.7))
