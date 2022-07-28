from mmcv import Config, DictAction
from decord import VideoReader
from mmaction.apis import init_recognizer
import os
import os.path as osp
import re
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from tqdm import tqdm
from mmaction.core import OutputHook
from mmaction.datasets.pipelines import Compose
from mmdatasets.const import get_root
from mmdatasets.datas import pick_datas
from lumo.utils import safe_io as IO
from lumo import Logger

log = Logger()
log.add_log_dir('./')


def feature_extraction_recognizer(model, video, outputs=None, as_tensor=True, **kwargs):
    """Inference a video with the recognizer.

    Args:
        model (nn.Module): The loaded recognizer.
        video (str | dict | ndarray): The video file path / url or the
            rawframes directory path / results dictionary (the input of
            pipeline) / a 4D array T x H x W x 3 (The input video).
        outputs (list(str) | tuple(str) | str | None) : Names of layers whose
            outputs need to be returned, default: None.
        as_tensor (bool): Same as that in ``OutputHook``. Default: True.

    Returns:
        dict[tuple(str, float)]: Top-5 recognition result dict.
        dict[torch.tensor | np.ndarray]:
            Output feature maps from layers specified in `outputs`.
    """

    input_flag = None
    if isinstance(video, dict):
        input_flag = 'dict'
    elif isinstance(video, np.ndarray):
        assert len(video.shape) == 4, 'The shape should be T x H x W x C'
        input_flag = 'array'
    elif isinstance(video, str) and video.startswith('http'):
        input_flag = 'video'
    elif isinstance(video, str) and osp.exists(video):
        if osp.isfile(video):
            if video.endswith('.npy'):
                input_flag = 'audio'
            else:
                input_flag = 'video'
        if osp.isdir(video):
            input_flag = 'rawframes'
    else:
        raise RuntimeError('The type of argument video is not supported: '
                           f'{type(video)}')

    if isinstance(outputs, str):
        outputs = (outputs,)
    assert outputs is None or isinstance(outputs, (tuple, list))

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline
    # Alter data pipelines & prepare inputs
    if input_flag == 'dict':
        data = video
    if input_flag == 'array':
        modality_map = {2: 'Flow', 3: 'RGB'}
        modality = modality_map.get(video.shape[-1])
        data = dict(
            total_frames=video.shape[0],
            label=-1,
            start_index=0,
            array=video,
            modality=modality)
        for i in range(len(test_pipeline)):
            if 'Decode' in test_pipeline[i]['type']:
                test_pipeline[i] = dict(type='ArrayDecode')
        test_pipeline = [x for x in test_pipeline if 'Init' not in x['type']]
    if input_flag == 'video':
        data = dict(filename=video, label=-1, start_index=0, modality='RGB')
        if 'Init' not in test_pipeline[0]['type']:
            test_pipeline = [dict(type='OpenCVInit')] + test_pipeline
        else:
            test_pipeline[0] = dict(type='OpenCVInit')
        for i in range(len(test_pipeline)):
            if 'Decode' in test_pipeline[i]['type']:
                test_pipeline[i] = dict(type='OpenCVDecode')
    if input_flag == 'rawframes':
        filename_tmpl = cfg.data.test.get('filename_tmpl', 'img_{:05}.jpg')
        modality = cfg.data.test.get('modality', 'RGB')
        start_index = cfg.data.test.get('start_index', 1)

        # count the number of frames that match the format of `filename_tmpl`
        # RGB pattern example: img_{:05}.jpg -> ^img_\d+.jpg$
        # Flow patteren example: {}_{:05d}.jpg -> ^x_\d+.jpg$
        pattern = f'^{filename_tmpl}$'
        if modality == 'Flow':
            pattern = pattern.replace('{}', 'x')
        pattern = pattern.replace(
            pattern[pattern.find('{'):pattern.find('}') + 1], '\\d+')
        total_frames = len(
            list(
                filter(lambda x: re.match(pattern, x) is not None,
                       os.listdir(video))))
        data = dict(
            frame_dir=video,
            total_frames=total_frames,
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
        if 'Init' in test_pipeline[0]['type']:
            test_pipeline = test_pipeline[1:]
        for i in range(len(test_pipeline)):
            if 'Decode' in test_pipeline[i]['type']:
                test_pipeline[i] = dict(type='RawFrameDecode')
    if input_flag == 'audio':
        data = dict(
            audio_path=video,
            total_frames=len(np.load(video)),
            start_index=cfg.data.test.get('start_index', 1),
            label=-1)

    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with OutputHook(model, outputs=outputs, as_tensor=as_tensor) as h:
        with torch.no_grad():
            scores = model(return_loss=False, **data)[0]
        returned_features = h.layer_outputs if outputs else None

    return scores, returned_features


from lumo import Params


class VFeatParams(Params):

    def __init__(self):
        super().__init__()
        self.config = '/Users/yhz/Documents/Python/mmaction2/configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py'
        self.checkpoint = '/Users/yhz/Documents/Python/mmaction2/checkpoints/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth'
        self.dataset = self.choice(
            'iemocap-cogmen-video-4',
            'iemocap-cogmen-video-6',
            'meld-mmgcn-video-6',
        )
        self.cfg_options = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.prefix = ''
        self.suffix = ''

    def iparams(self):
        super().iparams()
        self.prefix = os.path.basename(self.config).split('_')[0]


def main():
    # assign the desired device.

    pm = VFeatParams()
    pm.from_args()
    pm.iparams()
    print(pm)

    device = torch.device(pm.device)
    cfg = Config.fromfile(pm.config)
    cfg.merge_from_dict(pm.cfg_options)

    root = get_root(pm.dataset.split('-')[0])
    new_res = {}
    video_clips = pick_datas(root, pm.dataset, 'train')

    # build the recognizer from a config file and checkpoint file/url
    # hack for feature extraction
    model = init_recognizer(cfg, pm.checkpoint, device=device)
    model.feature_extraction = True

    new_res = {}
    uid = None
    for k, sample in tqdm(list(video_clips.items())):
        vr = VideoReader(sample['fn'])
        fps = vr.get_avg_fps()
        f_left = 'F' in os.path.basename(sample['fn'])
        features = []
        try:
            for (uid, left, right), gender, sents, lb in zip(sample['timestamp'], sample['speaker'],
                                                             sample['video_sentence'], sample['video_labels']):
                left_f, right_f = round(left * fps), round(right * fps)

                cur_left = f_left == gender
                arr = vr[left_f:right_f].asnumpy()
                if cur_left:
                    arr = arr[:, 120:365, :arr.shape[2] // 2]
                else:
                    arr = arr[:, 120:365, arr.shape[2] // 2:]

                results, _ = feature_extraction_recognizer(model, arr)
                features.append(results)
        except KeyboardInterrupt as e:
            raise e
        except:
            log.info(sample['fn'], k, uid)
            new_res[k] = []
            continue

        features = np.stack(features)
        new_res[k] = features

    if pm.dataset == 'iemocap-cogmen-video-6':
        IO.dump_pkl(new_res, os.path.join(root, 'cogmen', 'iemocap', f'{pm.prefix}_vfeat{pm.suffix}.pkl'))
    elif pm.dataset == 'iemocap-cogmen-video-4':
        IO.dump_pkl(new_res, os.path.join(root, 'cogmen', 'iemocap_4', f'{pm.prefix}_vfeat{pm.suffix}.pkl'))
    elif pm.dataset == 'meld-mmgcn-video-7':
        IO.dump_pkl(new_res, os.path.join(root, 'mmgcn', f'{pm.prefix}_vfeat{pm.suffix}.pkl'))


if __name__ == '__main__':
    main()
