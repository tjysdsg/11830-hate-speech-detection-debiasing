import pandas as pd
import os


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('pred', type=str, help='Path to the prediction file')
    parser.add_argument('out', type=str, help='Path to store the output csv')
    return parser.parse_args()


def main():
    args = get_args()

    ref = pd.read_csv(os.path.join('data', 'toxigen_annotated_test.csv'))
    target_groups = list(ref['target_group'].unique())
    # print(f'Target groups:')
    # print("\n".join(target_groups))
    # print("\n")

    ref['label'] = ((ref.toxicity_ai + ref.toxicity_human) > 5.5).astype(int)
    labels = list(ref['label'])

    preds_list = []
    with open(args.pred, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip().split('\t')
            # preds_list.append(int(line[0]))

            logits = line[1].replace('[', '').replace(']', '').split()
            logits = [float(x) for x in logits]
            assert len(logits) == 2

            if logits[0] > logits[1]:
                preds_list.append(0)
            else:
                preds_list.append(1)
    assert len(preds_list) == len(labels)

    df = pd.DataFrame({
        'identity': ref['target_group'],
        'label': ref['label'],
        'pred': preds_list,
    })

    res = {
        'identity': target_groups,
        'fp': [],
        'tn': [],
        'fpr': [],
    }
    for tg in target_groups:
        fp = 0
        tn = 0
        id_data = df.loc[df['identity'] == tg]  # data belonging this target group

        for i, row in id_data.iterrows():
            if row['label'] == 0 and row['pred'] == 1:
                fp += 1
            if row['label'] == 0 and row['pred'] == 0:
                tn += 1

        fpr = fp / (fp + tn)
        res['fp'].append(fp)
        res['tn'].append(tn)
        res['fpr'].append(fpr)

        # print(f"{tg} with FPR(fp/tn) = {fpr}({fp}/{tn})")

    res = pd.DataFrame(res)
    res.to_csv(args.out)


if __name__ == '__main__':
    main()
