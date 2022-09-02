import torch
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from lumo.utils import safe_io as IO

from scipy.io import wavfile
import os
from joblib import Parallel, delayed
from mmdatasets.datas import pick_datas
from mmdatasets.const import get_root
from lumo import Params
from transformers.models.roberta import RobertaModel, RobertaTokenizer


class FParams(Params):

    def __init__(self):
        super().__init__()
        self.mfcc = False
        self.stft = False
        self.fb = False

        self.dataset = self.choice(
            'meld-mmgcn-text-7',
            'iemocap-cogmen-text-6',
            'iemocap-cogmen-text-4',
            'mosei-text-any-7',
        )
        self.bert_type = self.choice('sbert', 'robert')
        self.pretrained_name = 'roberta-large'
        self.root = True
        self.merge = True

        self.device = 0 if torch.cuda.is_available() else 'cpu'

    def iparams(self):
        super().iparams()


def extract(model, t):
    res = model.forward(**t)
    return res.pooler_output.cpu().numpy()


def extract_sbert(model, t):
    return model.encode(t, show_progress_bar=False)


def main():
    pm = FParams()

    pm.from_args()
    pm.iparams()

    print(pm)

    root = get_root(pm.dataset.split('-')[0])
    new_res = {}
    video_sentence = pick_datas(root, pm.dataset, 'train')

    if pm.bert_type == 'sbert':
        model = SentenceTransformer("paraphrase-distilroberta-base-v1")
        model.to(pm.device)
        for k in tqdm(video_sentence):
            new_res[k] = extract_sbert(model, video_sentence[k])
    else:
        model = RobertaModel.from_pretrained(pm.pretrained_name)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model.eval()
        model.to(pm.device)
        with torch.no_grad():
            for k in tqdm(video_sentence):
                new_res[k] = extract(model,
                                     tokenizer(video_sentence[k], padding=True, return_tensors='pt').to(pm.device))

    if pm.dataset == 'iemocap-cogmen-text-6':
        fn = os.path.join(root, 'cogmen', 'iemocap', f'{pm.bert_type}_map.pkl')
        IO.dump_pkl(new_res, fn)
    elif pm.dataset == 'iemocap-cogmen-text-4':
        fn = os.path.join(root, 'cogmen', 'iemocap_4', f'{pm.bert_type}_map.pkl')
        IO.dump_pkl(new_res, fn)
    elif pm.dataset == 'meld-mmgcn-text-7':
        fn = os.path.join(root, 'mmgcn', f'{pm.bert_type}_map.pkl')
        IO.dump_pkl(new_res, fn)
    elif pm.dataset == 'mosei-text-any-7':
        fn = os.path.join(root, f'{pm.bert_type}_map.pkl')
        IO.dump_pkl(new_res, fn)
    else:
        raise NotImplementedError(pm.dataset)
    print(fn)


if __name__ == '__main__':
    main()
