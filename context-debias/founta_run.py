import pandas as pd
import torch
import numpy as np

FILTER_LIST = ["RT"]


def remove_user_link(text):
    text_list = text.split(" ")
    clean_text = [t for t in text_list if "@" not in t and "https:" not in t and "#" not in t]
    clean_text = [t for t in clean_text if t not in FILTER_LIST]
    return " ".join(t for t in clean_text)


def add_cleantext(filename):
    df = pd.read_csv("data/{}.csv".format(filename))
    df.loc[:, "CleanText"] = df.Text.apply(remove_user_link)
    df = df.dropna(subset=['CleanText'])
    df = df[df.CleanText != ""]
    df.to_csv("data/{}_clean.csv".format(filename))


add_cleantext("founta_valid")
add_cleantext("founta_train")
add_cleantext("founta_test")
