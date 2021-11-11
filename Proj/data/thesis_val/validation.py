from sklearn.metrics.pairwise import cosine_similarity
from my_utils import word_vec, get_keybert_model
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

MEAN_THD = 999

model = {
    "keyBert": "distilbert-base-nli-mean-tokens",
    "keyBert_mpnet": "paraphrase-mpnet-base-v2",
}

model_keybert = get_keybert_model(model["keyBert_mpnet"])

p_data = "./data"

test_pipline = [
    # "keywords_keyBert_mpnet",
    # "keywords_keyBert_mpnet3",
    # "keywords_keyBert",
    "keywords_keyBert3",
    "keywords_yake",
    "keywords_yake3",
]

for test_proj in test_pipline:
    f_proj_p = "{}/thesis/{}.pickle".format(p_data, test_proj)
    f_val_p = "{}/thesis_val/{}.pickle".format(p_data, test_proj)
    f_val_excel_p = "{}/thesis_val/{}.xlsx".format(p_data, test_proj)

    print("# In proj: {}...".format(test_proj))

    with open(f_proj_p, "rb") as f:
        df_proj: pd.DataFrame = pickle.load(f)

    l_line_acc = []
    l_line_len_diff = []
    for index in tqdm(df_proj.index):
        l_kw_g = df_proj.loc[index, "KWG"]
        l_kw_t = df_proj.loc[index, "KW"]
        len_diff = len(l_kw_g) - len(l_kw_t)
        l_line_len_diff.append(len_diff)
        if (len(l_kw_g) == 1 and not isinstance(l_kw_g[0], tuple)) or len(l_kw_g) == 0:
            line_acc = 0
        else:
            if len_diff > 0:
                l_kw_g = l_kw_g[: len(l_kw_t)]
            l_kw_vec_g = [word_vec(i[0], model_keybert.model) for i in l_kw_g]
            l_kw_vec_t = [word_vec(i, model_keybert.model) for i in l_kw_t]
            # 行是生成的某关键词，列是原有的
            sim_matrix = np.zeros((len(l_kw_g), len(l_kw_t)))
            for i, kw_vec_g in enumerate(l_kw_vec_g):
                for j, kw_vec_t in enumerate(l_kw_vec_t):
                    sim = cosine_similarity(kw_vec_g, kw_vec_t)[0][0]
                    sim_matrix[i][j] = sim

            # 选择匹配关系
            l_kwg_acc = np.zeros(len(l_kw_g))
            max_num_idx = sim_matrix.flatten().argsort()
            max_num_idx = [
                (i_pos // len(l_kw_t), i_pos % len(l_kw_t)) for i_pos in max_num_idx
            ]
            used_pos = {"i_kwg": set(), "i_kw": set()}
            for i in range(len(l_kw_g)):
                i_pos = max_num_idx.pop(-1)
                if (
                    i_pos[0] not in used_pos["i_kwg"]
                    and i_pos[1] not in used_pos["i_kw"]
                ):
                    used_pos["i_kwg"].add(i_pos[0])
                    used_pos["i_kw"].add(i_pos[1])
                    acc = sim_matrix[i_pos[0]][i_pos[1]]
                    l_kwg_acc[i_pos[0]] = acc
                    # print("KWG {} -> KW {} ({:.4f})".format(i_pos[0], i_pos[1], acc))
            if len(l_kwg_acc) > MEAN_THD:
                line_acc = np.mean(l_kwg_acc[np.argsort(l_kwg_acc)[-MEAN_THD:]])
            else:
                line_acc = np.mean(l_kwg_acc)
            # line_acc = np.mean(l_kwg_acc)
        l_line_acc.append(line_acc)
    df_proj["LEN_DIFF"] = l_line_len_diff
    df_proj["ACC"] = l_line_acc

    acc_proj = np.mean(l_line_acc)
    print("acc: {:.5f}".format(acc_proj))

    with open(f_val_p, "wb") as f:
        pickle.dump(df_proj, f)

    df_proj.to_excel(f_val_excel_p, index=False)
