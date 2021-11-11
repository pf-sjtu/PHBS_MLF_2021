#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from my_utils import get_keybert_model, model_name, word_vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
from adjustText import adjust_text
from datetime import timedelta, datetime
from tqdm import tqdm
import math
from scipy import interpolate
import matplotlib.cm as cmx
import matplotlib.colors as colors
from astroML.plotting import scatter_contour

# In[2]:
p_data = "./data"
porj_name = "keywords_keyBert"
f_patent_keybert_p = "{}/patents/{}.pickle".format(p_data, porj_name)
f_patent_keybert_vec_p = "{}/patents/{}_vec.pickle".format(p_data, porj_name)
f_patent_keybert_clust_p = "{}/patents/{}_clust.pickle".format(p_data, porj_name)
f_patent_keybert_period_p = "{}/patents/{}_period.pickle".format(p_data, porj_name)

LOAD_DATA = True


# !!!
CLIP_THD = 0.5  # 3
PLOT_THD = 5
N_CLUST = 5
N_COMM_KW = 5
YEAR_BEG = 1990
YEAR_INT = 5
YEAR_END = 2022
HEAT_BINS = 30
SIM_DOTS_IMPS_INT = 0.5

# In[3]:
with open(f_patent_keybert_p, "rb") as f:
    df_patent_keybert = pickle.load(f)


def kwg_agg_func(l):
    l = [i[0] for i in l if isinstance(i, tuple)]
    return ", ".join(l)


def kwg_agg_func2(l):
    l = ["{:.5f}".format(i[1]) for i in l if isinstance(i, tuple)]
    return ", ".join(l)


f_patent_keybert_excel_p = "{}/patents/{}.xlsx".format(p_data, porj_name)
df_patent_keybert2 = df_patent_keybert.copy()
df_patent_keybert2["KWG"] = df_patent_keybert["KWG"].apply(kwg_agg_func)
df_patent_keybert2["importance"] = df_patent_keybert["KWG"].apply(kwg_agg_func2)
df_patent_keybert2.to_excel(f_patent_keybert_excel_p, index=False)
# In[5]:
def cleaning(df, col="KWG"):
    re_rm = r".*(research|also|method|way|invention|generation).*"
    re_rm_sub = r"^\d+ | \d+$|applying|implementation|implementated|applies|apply|including|includes|include|the |an |a |some |have |had |another |other |more |each |first |second |last"
    re_rm_sub2 = r"^ +| +$"
    re_rm_empty = r"^[^a-zA-Z]*$"

    df = df[df[col].apply(lambda x: re.match(re_rm, x) is None)]
    df[col] = df[col].apply(lambda x: re.sub(re_rm_sub, "", x))
    df[col] = df[col].apply(lambda x: re.sub(re_rm_sub2, "", x))
    df[col] = df[col].apply(lambda x: re.sub(r" +", " ", x))
    df = df[df[col].apply(lambda x: re.match(re_rm_empty, x) is None)]

    return df


def gen_df_kws(df, col="KWG"):
    keypairs = df["KWG"].sum()
    keypairs = pd.DataFrame(keypairs, columns=["KW", "importance"])
    keypairs_sum = keypairs.groupby("KW").sum().reset_index()
    keypairs_sum = cleaning(keypairs_sum, "KW")
    return keypairs_sum

    # In[]:


l_period = [datetime(YEAR_BEG, 1, 1)]
dt_end = datetime(YEAR_END, 1, 1)
l_df_period = []
while l_period[-1] < dt_end:
    l_period.append(datetime(l_period[-1].year + YEAR_INT, 1, 1))
    flags = (df_patent_keybert["publication_date"] >= l_period[-2]) & (
        df_patent_keybert["publication_date"] < l_period[-1]
    )
    df = df_patent_keybert[flags]
    if df.shape[0] > 0:
        l_df_period.append(
            {
                "dt_beg": l_period[-2],
                "dt_end": l_period[-1],
                "df": df,
                "df_kws": gen_df_kws(df, "KWG"),
                "n": df.shape[0],
            }
        )

if not LOAD_DATA:
    # In[5]:
    keypairs_sum = gen_df_kws(df_patent_keybert, "KWG")

    # In[6]: Distribution of keywords' importances
    imp = keypairs_sum["importance"]
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].hist(imp, bins=30, log=True)
    ax[1].hist(imp[imp > CLIP_THD], bins=30, log=False)
    ax[0].set_ylabel("Freq")
    ax[1].set_ylabel("Freq")
    ax[0].set_xlabel("Importance")
    ax[1].set_xlabel(
        "Importance (>{:.0f})".format(CLIP_THD) if CLIP_THD > 0 else "Importance"
    )
    ax[0].set_title("Distribution of keywords' importances")
    fig.savefig("{}/plot/dist_of_imp.pdf".format(p_data), bbox_inches="tight")
    fig.show()

    # In[7]:
    df_kws = keypairs_sum[keypairs_sum["importance"] > CLIP_THD]

    # In[ ]:
    model_keybert = get_keybert_model(model_name["keyBert_mpnet"])

    # In[ ]: 获得每个关键词的词向量
    pbar = tqdm(total=df_kws.shape[0])

    def vec_fun(s):
        global model_keybert, tqdm
        wv = word_vec(s, model_keybert.model)
        pbar.update(1)
        return wv

    df_kws["vec"] = df_kws["KW"].apply(vec_fun)
    pbar.close()

    with open(f_patent_keybert_vec_p, "wb") as f:
        pickle.dump(df_kws, f)

else:
    with open(f_patent_keybert_vec_p, "rb") as f:
        df_kws = pickle.load(f)

# In[]:
pca = PCA(n_components=2)  # 降到2维
X = np.stack(df_kws["vec"])[:, 0, :]  # 导入数据，维度为768
pca.fit(X)
X2 = pca.fit_transform(X)  # 降维后的数据放在newX里
df_kws["X1"] = X2[:, 0]
df_kws["X2"] = X2[:, 1]

# In[]:
def get_cmap(N):
    """Returns a function that maps each index in 0, 1,.. . N-1 to a distinct
    RGB color."""
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap="Set2")
    # scalar_map = cmx.ScalarMappable(norm=color_norm, cmap="hsv")

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color


idx2color = get_cmap(N_CLUST + 1)

# In[ ]:
# if not LOAD_DATA:
kmeans = KMeans(n_clusters=N_CLUST, random_state=0).fit(X2)
centers = kmeans.cluster_centers_
print("Centers: {}".format(centers))

# In[ ]:
l_clust = []
df_kws["clust"] = -1
for i in range(N_CLUST):
    df_kws_clust = df_kws.iloc[kmeans.labels_ == i].sort_values(
        "importance", ascending=False
    )
    df_comm_kw = df_kws_clust.iloc[:N_COMM_KW, :]
    l_comm_idx = df_kws_clust.index
    df_kws.loc[l_comm_idx, "clust"] = i
    s_comm_kw = ", ".join(df_comm_kw["KW"])
    l_clust.append(
        {
            "i": i,
            "center": kmeans.cluster_centers_[i],
            "n": df_kws_clust.shape[0],
            "df_kws_clust": df_kws_clust,
            "df_comm_kw": df_comm_kw,
            "l_comm_idx": l_comm_idx,
            "s_comm_kw": s_comm_kw,
        }
    )

# In[]:
merge_col = [i for i in df_kws.columns if i != "importance"]
for i in range(len(l_df_period)):
    df_p_kws = l_df_period[i]["df_kws"]
    l_df_period[i]["df_kws"] = df_p_kws.merge(df_kws[merge_col], how="left", on="KW")

# In[]:
for i in range(len(l_df_period)):
    df_p_kws = l_df_period[i]["df_kws"]
    l_df_period[i]["df_kws"] = df_p_kws[~df_p_kws["KW"].duplicated()]
    df_p_kws = l_df_period[i]["df_kws"]
    sum_imp = df_p_kws["importance"].sum()
    weight_x1 = (df_p_kws["X1"] * df_p_kws["importance"]).sum() / sum_imp
    weight_x2 = (df_p_kws["X2"] * df_p_kws["importance"]).sum() / sum_imp
    l_df_period[i]["pos_weight"] = (weight_x1, weight_x2)

# In[]:
# if not LOAD_DATA:
# with open(f_patent_keybert_vec_p, "wb") as f:
#     pickle.dump(df_kws, f)
# with open(f_patent_keybert_clust_p, "wb") as f:
#     pickle.dump(l_clust, f)
# with open(f_patent_keybert_period_p, "wb") as f:
#     pickle.dump(l_df_period, f)

# # In[]:
# else:
# with open(f_patent_keybert_vec_p, "rb") as f:
#     df_kws = pickle.load(f)
# with open(f_patent_keybert_clust_p, "rb") as f:
#     l_clust = pickle.load(f)
# with open(f_patent_keybert_period_p, "rb") as f:
#     l_df_period = pickle.load(f)


# In[ ]:

from my_utils import plot_kws, get_sim_dots


# In[ ]:

# scatter
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
df_plot_kws = df_kws[df_kws["importance"] > PLOT_THD]
plot_kws(
    fig,
    ax,
    color_func=idx2color,
    df_plot_kws=df_plot_kws,
    bins=HEAT_BINS,
    a_sim_dots=None,
    n_comm_kws=N_COMM_KW,
    l_clust=l_clust,
    l_df_period=None,
    year_int=YEAR_INT,
    period_arrow=False,
    period_line=False,
    kw_color_func_flag=False,
    p_data=p_data,
    file_name="scatter",
)

# heat
a_sim_dots = get_sim_dots(df_kws=df_kws, sim_dots_imp_int=SIM_DOTS_IMPS_INT)
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
plot_kws(
    fig2,
    ax2,
    color_func=idx2color,
    df_plot_kws=None,
    bins=HEAT_BINS,
    a_sim_dots=a_sim_dots,
    n_comm_kws=N_COMM_KW,
    l_clust=l_clust,
    l_df_period=l_df_period,
    year_int=YEAR_INT,
    period_arrow=False,
    period_line=True,
    kw_color_func_flag=True,
    p_data=p_data,
    file_name="heat",
)

# In[]:
# yearly heat
df_xy_nonan = df_kws[["X1", "X2"]].dropna()
x_range = [df_xy_nonan["X1"].min(), df_xy_nonan["X1"].max()]
y_range = [df_xy_nonan["X2"].min(), df_xy_nonan["X2"].max()]
for i in range(len(l_df_period)):
    if i in [0]:
        continue
    df_kws_p = l_df_period[i]["df_kws"]
    a_sim_dots_p = get_sim_dots(df_kws=df_kws_p, sim_dots_imp_int=SIM_DOTS_IMPS_INT)

    n_dt_beg = l_df_period[i]["dt_beg"].year
    if i < len(l_df_period) - 1:
        s_year = "{}-{}".format(n_dt_beg, n_dt_beg + YEAR_INT)
    else:
        s_year = "{}-now".format(n_dt_beg)

    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 10))
    ax3.set_xlim(x_range[0], x_range[1])
    ax3.set_ylim(y_range[0], y_range[1])
    plot_kws(
        fig3,
        ax3,
        color_func=None,
        df_plot_kws=None,
        bins=7,
        a_sim_dots=a_sim_dots_p,
        n_comm_kws=N_COMM_KW,
        l_clust=[{"df_kws_clust": df_kws_p.sort_values("importance", ascending=False)}],
        l_df_period=None,
        year_int=YEAR_INT,
        period_arrow=False,
        period_line=False,
        kw_color_func_flag=False,
        p_data=p_data,
        file_name="heat_" + s_year,
    )


# In[]:
f_patent_simp_clust_p = "{}/patents/sim_{}_clust.xlsx".format(p_data, porj_name)
f_patent_simp_period_p = "{}/patents/sim_{}_period.xlsx".format(p_data, porj_name)

df_kws[["KW", "importance", "X1", "X2", "clust"]].to_excel(
    f_patent_simp_clust_p, index=False
)

for i in range(len(l_df_period)):
    l_df_period[i]["df_kws"]["dt_beg"] = l_df_period[i]["dt_beg"].year

df_period_all = pd.concat([i["df_kws"] for i in l_df_period])
df_period_all[["KW", "importance", "X1", "X2", "clust", "dt_beg"]].to_excel(
    f_patent_simp_period_p, index=False
)
