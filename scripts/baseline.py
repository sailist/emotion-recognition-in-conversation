"""
python3 scripts/dagerc.py | bash
"""
import torch

from lumo import Params, Logger
from lumo.utils.fmt import strftime

log = Logger()
log.use_stdout = False
fn = log.add_log_dir(f'./run_baseline')


def main(module=None, modality='1111111', n_classes='46', script='train_mm.py'):
    pm = Params()

    pm.gpus = torch.cuda.device_count()
    pm.module = module
    pm.modality = modality
    pm.seeds = 3
    pm.script = script
    pm.from_args()
    pm.modality = str(pm.modality)

    log.raw(pm)

    base = """python3 {script} --module={module} --dataset={dataset} --reimplement --modality={modality} --seed={seed} --device={device} --baseline & \n"""

    device = 0

    all_modality = ['atv', 'av', 'at', 'tv', 'a', 't', 'v']
    # all_feature_type = ['iemocap-cogmen-sbert-4', 'iemocap-cogmen-sbert-6', 'meld-mmgcn-sbert-7', 'meld-mmgcn-7']
    all_feature_type = ['mosei-emo-sbert-fbank-6', 'mosei-sent-sbert-fbank-2', 'mosei-sent-sbert-fbank-7']
    sh = []
    for seed in range(pm.seeds):
        for modality in [m for i, m in enumerate(all_modality) if
                         int(pm.modality.ljust(len(all_modality), '0')[i]) == 1]:
            for dataset in all_feature_type:
                dataset = dataset.format(n_classes)
                cur = base.format(
                    script=pm.script,
                    seed=seed, modality=modality,
                    device=device, dataset=dataset, module=pm.module)

                if pm.gpus == 0:
                    device = 'cpu'
                else:
                    device = (device + 1) % pm.gpus
                sh.append(cur)

    print(f'echo "execute {len(sh)} tests."')

    step = pm.gpus if pm.gpus > 0 else 1
    for i in range(0, len(sh), step):
        cmds = sh[i:i + step]
        cmds = [f'sleep {i * 2}; echo "{strftime()} {cmd}" >> {fn} ; {cmd}' for i, cmd in enumerate(cmds)]
        print(''.join(cmds) + 'wait')


if __name__ == '__main__':
    main('mmgcn', modality='1111000')
    main('iemocap')
    main('dagerc')
