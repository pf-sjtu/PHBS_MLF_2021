# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 16:41:32 2021

@author: PENG Feng
@email:  im.pengf@outlook.com
"""

import pandas as pd
import os
import pickle

p_data = "./data"
f_patent_all_p = "{}/patents/google_patents.pickle".format(p_data)
f_patent_clean_p = "{}/patents/google_patents_clean.pickle".format(p_data)
f_thesis_p = "{}/thesis/science_direct.pickle".format(p_data)

if __name__ == "__main__":
    AGG_CSV = False
    CLEAN_PICKLE = False
    CLEAN_THESIS_ABS = False

    if AGG_CSV:
        f_patents = os.listdir(p_data)
        f_patent_all = "{}/patents/google_patents.csv".format(p_data)
        f_patent_all_p = "{}/patents/google_patents.pickle".format(p_data)

        s_patent_all = ""
        for f_patent in f_patents:
            f_patent = "{}/{}".format(p_data, f_patent)
            with open(f_patent, "r", encoding="utf-8") as f:
                s_patent = f.read().strip("\n")
                s_patent_all += "\n" + s_patent

        with open(f_patent_all, "w", encoding="utf-8") as f:
            f.write(s_patent_all)

        patent_df = pd.read_csv(f_patent_all)

        with open(f_patent_all_p, "wb") as f:
            pickle.dump(patent_df)

    if CLEAN_PICKLE:
        with open(f_patent_all_p, "rb") as f:
            patent_df = pickle.load(f)

        def patent_filter(line):
            i = {
                "title": 0,
                "publication_date": 1,
                "application_number": 2,
                "assignees": 3,
                "cpc": 4,
                "abstract": 5,
            }
            flag = line[i["application_number"]][:2] == "US"
            return flag

        patent_clean_df = patent_df[patent_df.apply(patent_filter, axis=1)]

        def cpc_agg(df):
            s = "; ".join(df["cpc"])
            df = df.iloc[0, :]
            df["cpc"] = s
            return df

        patent_clean_df = patent_clean_df.groupby("application_number").apply(cpc_agg)
        patent_clean_df = patent_clean_df.reset_index(drop=True)

        def assignees_clean(s):
            s2 = (
                s.replace(",Inc", ", Inc")
                .replace(", Inc", " Inc")
                .replace(",Llc", ", Llc")
                .replace(", Llc", " Llc")
                .replace(", a", "| a")
                .replace(", A", "| a")
                .replace(",", ";")
                .replace("|", ",")
                .replace("  ", " ")
                .replace(" ,", ",")
                .replace("Delware", "Delaware")
            )
            return s2

        t = patent_clean_df["assignees"]
        patent_clean_df["assignees"] = t[t.apply(lambda x: ", " in x)]

        patent_clean_df["publication_date"] = pd.to_datetime(
            patent_clean_df["publication_date"], format="%Y%m%d"
        )

        with open(f_patent_clean_p, "wb") as f:
            pickle.dump(patent_clean_df, f)

    if CLEAN_THESIS_ABS:
        f_thesis = "{}/thesis/science_direct.xlsx".format(p_data)
        df_thesis = pd.read_excel(f_thesis)

        df_thesis = df_thesis.iloc[:, 1:]
        df_thesis["KW"] = (
            df_thesis["KW"].apply(lambda x: x.replace(";", "; ")).reset_index(drop=True)
        )

        with open(f_thesis_p, "wb") as f:
            pickle.dump(df_thesis, f)

else:
    with open(f_patent_clean_p, "rb") as f:
        df_patent = pickle.load(f)
    with open(f_thesis_p, "rb") as f:
        df_thesis = pickle.load(f)
