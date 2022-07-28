import json
import os
import traceback
from datetime import datetime
from typing import Any

from lumo import callbacks, Trainer, ParamsType
from lumo.proc.dist import world_size
from lumo.core import BaseParams
from lumo.proc.dist import is_main


class NotionCallback(callbacks.TrainCallback, callbacks.InitialCallback):
    class EmptyObject:
        def __getattr__(self, item):
            return self.__call__

        def __setattr__(self, key, value):
            pass

        def __call__(self, *args, **kwargs):
            pass

    def ensure_properties(self):
        from potion.objects import sche

        # TODO pop exists same type properties

        res, resp = self.database.update_properies([
            sche.RichText('Experiment Name'),
            sche.Date('Duration'),
            sche.RichText('LogFile'),
            sche.RichText('Params Hash'),
            sche.Number('GPU number', format='number'),
            sche.RichText('model'),
            # sche.MultiSelect('dataset', []),
            sche.Number('batch_size', format='number'),
            sche.Number('epoch', format='number'),
            sche.Number('Result', format='number'),
            sche.RichText('seed'),
            sche.CheckBox('ema'),
            # sche.MultiSelect('optimizer', []),
            # sche.MultiSelect('optimizer.lr', []),
            # sche.MultiSelect('optimizer.weight_decay', []),
            sche.RichText('Arg String'),
            sche.RichText('Key Arg'),
            sche.RichText('Kill'),
            sche.RichText('Commit Hex'),
            # sche.MultiSelect('Status', []),
            sche.Date('End'),
            sche.Number('Running Time', format='number'),
            sche.CheckBox('Finished'),
        ])
        return res, resp

    @property
    def npage(self):
        if self._npage is None:
            return self.EmptyObject()
        return self._npage

    def on_hooked(self, source: Trainer, params: ParamsType):
        super(NotionCallback, self).on_hooked(source, params)
        self._npage = None
        self.start = datetime.now().astimezone()
        if params.get('notion_page_id', None) is None:
            source.logger.info('notion_page_id not in params.')
            return
        if not is_main():
            return
        from potion.beans import NotionDatabase

        self.database = NotionDatabase(id=params.notion_page_id)
        # self.database.object.properties
        self.ensure_properties()

        try:
            self._npage = self.database.create_page()
        except Exception as e:
            source.logger.info(f'{type(e)}{e} happends when create NotionPage.')
            return

        source.exp.dump_info('notion', {
            'type': 'page_id',
            'page_id': self.npage.id
        })

        self.npage.set_title(source.exp.test_name)
        self.npage.set_text('Experiment Name', source.exp.exp_name)

        self.npage.set_duration('Duration', start=self.start)

        source.exp.add_exit_hook(self.finished)

        self.npage.set_text('LogFile', '\n'.join(source.logger.out_channel))

        self.npage.set_text('Params Hash', params.hash())

        self.npage.set_number('GPU number', world_size())

        md = params.get('model', None)
        if md is not None:
            self.npage.set_text('model', md)

        ds = params.get('dataset', None)
        if ds is not None:
            self.npage.add_option('dataset', ds)

        bs = params.get('batch_size', None)
        if bs is not None:
            self.npage.set_number('batch_size', bs)

        epoch = params.get('epoch', None)
        if epoch is not None:
            self.npage.set_number('epoch', epoch)

        seed = params.get('seed', None)
        if seed is not None:
            self.npage.set_text('seed', seed)

        ema = params.get('ema', None)
        if ema is not None:
            self.npage.set_checkbox('ema', ema)

        optim = params.get('optim', BaseParams())
        if optim is not None:
            self.npage.add_option('optimizer', optim.get('name', None))
            self.npage.add_option('optimizer.lr', optim.get('lr', None))
            self.npage.add_option('optimizer.weight_decay', optim.get('weight_decay', None))

        self.npage.set_text('Arg String', ' '.join(source.exp.exec_argv))
        self.npage.set_text('Key Arg', ' '.join([i for i in source.exp.exec_argv
                                                 if all([k not in i for k in {'seed', 'device'}])]))
        self.npage.set_text('Kill', f'kill -2 {os.getpid()}')
        self.npage.append_code(f'kill -2 {os.getpid()}', 'shell')
        self.npage.append_code('\n'.join(source.logger.out_channel), 'shell')
        self.npage.append_code(' '.join(source.exp.exec_argv), 'shell')
        self.npage.append_code(params.to_yaml(), language='yaml')

    def on_imodels_end(self, trainer: 'Trainer', func, params: ParamsType, result: Any, *args, **kwargs):
        super().on_imodels_end(trainer, func, params, result, *args, **kwargs)
        if not is_main():
            return

        if self.npage is None:
            return

        commit_info = trainer.exp.get_prop('git')
        if commit_info is not None and 'commit' in commit_info:
            self.npage.set_text('Commit Hex', commit_info['commit'])
        self.npage.append_code(json.dumps(commit_info, ensure_ascii=False, indent=2))
        self.npage.flush_children()
        self.npage.flush_property()

    def on_train_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        if self.npage is None:
            return
        self.npage.add_option('Status', 'train')

    def finished(self, *args):
        if self.npage is None:
            return
        end = datetime.now().astimezone()
        self.npage.set_duration('Duration', start=self.start, end=end)
        self.npage.set_duration('End', start=end)
        self.npage.set_number('Running Time', number=round((end - self.start).total_seconds() / 60, 2))
        self.npage.set_checkbox('Finished', True)
        self.npage.flush_property()
        self.npage.flush_children()

    def on_test_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        if self.npage is None:
            return
        self.npage.add_option('Status', 'test')

    def on_eval_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        if self.npage is None:
            return
        self.npage.add_option('Status', 'evaluation')

    def on_first_exception(self, source: Trainer, func, params: ParamsType, e: BaseException, *args, **kwargs):
        if self.npage is None:
            return

        self.npage.set_text('Exception', ''.join(traceback.format_exception_only(type(e), e)).split('\n')[0])

        self.npage.append_code(code=''.join(traceback.format_exception(type(e), e, e.__traceback__)),
                               language='plain text')
        self.npage.set_duration('Duration', start=self.start, end=datetime.utcnow().astimezone())
