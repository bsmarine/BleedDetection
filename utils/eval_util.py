import pandas as pd


def patient_based_filter(sub_df: pd.DataFrame):
    """Filtered by the required `class_label`, then output the highest score"""
    class_label = sub_df["class_label"].max()
    b_cand = sub_df[sub_df["class_label"] == class_label]
    return b_cand.loc[b_cand["pred_score"].idxmax()]
