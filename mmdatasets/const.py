from config import roots
try:
    from .datas import regist_data
except:
    from datas import regist_data



def get_root(dataset_name: str):
    return roots[dataset_name.split('-')[0]]


mean_std_dic = {
    'default': [(0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)],
}

imgsize_dic_ = {
    'ckplus': 48,
}


def imgsize_dic(dataset_name):
    res = dataset_name.split('-')
    assert res[0] in imgsize_dic_
    if len(res) == 2:
        return int(res[1])
    return imgsize_dic_[res[0]]


lazy_load_ds = {
    'ckplus'
}

n_classes = {
    k: round(float(k.split('-')[-1])) for k in regist_data
}
