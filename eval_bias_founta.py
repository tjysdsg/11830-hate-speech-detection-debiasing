import os
import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score


def get_scores(pred):
    """
    calculate relevant statistics for measuring bias.
    """
    prediction = pred['pred'].values.tolist()
    labels = pred['gt_label'].values.tolist()
    acc = accuracy_score(labels, prediction)
    f1 = f1_score(labels, prediction)

    fp = 0
    tn = 0
    n = len(prediction)
    assert n == len(labels)
    for i in range(n):
        if prediction[i] == 1 and labels[i] == 0:
            fp += 1
        elif prediction[i] == 0 and labels[i] == 0:
            tn += 1

    return [acc, f1, fp / (fp + tn), fp, tn]


def load_predictions(path: str) -> pd.DataFrame:
    df_dict = dict(
        gt_label=[],
        pred=[],
        # text=[],  # space-separated word sequence
    )

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n').split('\t')
            df_dict['gt_label'].append(int(line[0]))

            logits = line[1].replace('[', '').replace(']', '').split()
            logits = [float(x) for x in logits]
            assert len(logits) == 2
            pred = 0 if logits[0] > logits[1] else 1
            df_dict['pred'].append(pred)

            # df_dict['text'].append(line[2])

    return pd.DataFrame(df_dict)


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='data/founta_test.csv')
    parser.add_argument('--word-list', type=str, default='data/word_based_bias_list.csv')
    parser.add_argument('--prediction', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    return parser.parse_args()


def select_re_df(data, pred, regex):
    """
    select relevant dataframe to calculate statistics.
    """
    assert data.shape[0] == pred.shape[0]
    filter_ = data["Text"].apply(regex.findall).astype(bool)

    pred_new = pred[filter_].copy(deep=True)
    data_new = data[filter_]
    pred_new['text'] = data_new['Text']

    return pred_new


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    preds = load_predictions(args.prediction)
    data = pd.read_csv(args.data)

    # load biased word list
    df_word = pd.read_csv(args.word_list)
    noi_wordlist = df_word[df_word.categorization == 'harmless-minority'].word
    oi_wordlist = df_word[df_word.categorization == 'offensive-minority-reference'].word
    oni_wordlist = df_word[df_word.categorization == 'offensive-not-minority'].word

    # noi_wordlist.to_csv('noi_wordlist.csv', index=False)
    # oi_wordlist.to_csv('oi_wordlist.csv', index=False)
    # oni_wordlist.to_csv('oni_wordlist.csv', index=False)

    noi_regex = re.compile(r"\b" + r"\b|\b".join(noi_wordlist) + "\b", re.IGNORECASE)
    oi_regex = re.compile(r"\b" + r"\b|\b".join(oi_wordlist) + "\b", re.IGNORECASE)
    oni_regex = re.compile(r"\b" + r"\b|\b".join(oni_wordlist) + "\b", re.IGNORECASE)

    noi = select_re_df(data, preds, noi_regex)
    oi = select_re_df(data, preds, oi_regex)
    oni = select_re_df(data, preds, oni_regex)

    # noi.to_csv('noi.csv', index=False)
    # oi.to_csv('oi.csv', index=False)
    # oni.to_csv('oni.csv', index=False)

    noi_scores = get_scores(noi)
    oi_scores = get_scores(oi)
    oni_scores = get_scores(oni)
    scores_dict = dict(
        group=['noi', 'oi', 'oni'],
        acc=[noi_scores[0], oi_scores[0], oni_scores[0]],
        f1=[noi_scores[1], oi_scores[1], oni_scores[1]],
        fpr=[noi_scores[2], oi_scores[2], oni_scores[2]],
        fp=[noi_scores[3], oi_scores[3], oni_scores[3]],
        tn=[noi_scores[4], oi_scores[4], oni_scores[4]],
    )
    scores = pd.DataFrame(scores_dict)
    scores.to_csv(os.path.join(out_dir, 'founta_bias_eval.csv'), index=False)


if __name__ == '__main__':
    main()
