"""
ck+ dataset,
can be downloaded from https://www.kaggle.com/datasets/shawon10/ckplus
"""
import os

from sklearn.model_selection import train_test_split


def ckplus(root, split='train', fold=0):
    """
    :param root:
    :param split:
    :param fold:
    :return:
    """
    classes = sorted(os.listdir(root))
    classes = [i for i in classes if not i.startswith('.')]

    xs = []
    ys = []
    for i, cls in enumerate(classes):
        fs = os.listdir(os.path.join(root, cls))
        ycls = [i] * len(fs)
        fs = [i for i in fs if i.endswith('.png')]
        xs.extend([os.path.join(root, cls, f) for f in fs])
        ys.extend(ycls)
    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.1, random_state=fold)

    if split == 'train':
        return X_train, y_train
    else:
        return X_test, y_test


if __name__ == '__main__':
    xs, ys = ckplus('/Users/yhz/Downloads/ckplus/CK+48')
    print(len(xs))
