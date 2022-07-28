import json
import os
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Any

from lumo import callbacks, Trainer, ParamsType
from potion import Request
from potion.api import *
from potion.objects import Parent, Page, Properties
from potion.objects import block, rich, prop, Error as NotionError
from lumo.proc.dist import world_size
from lumo.core import BaseParams
from lumo.proc.dist import is_main


def walk_str(params, mem=''):
    for k, v in params.items():
        if not isinstance(v, dict):
            yield f"--{mem}{k}={v}"
        else:
            yield from walk_str(v, f'{k}.')


class RichTextProp(prop.Property):
    property_name = 'rich_text'

    def __init__(self, pname: str, rich_text):
        super().__init__(pname, args=rich_text)

    @staticmethod
    def PageTitle(rich_text):
        return prop.Title(None, rich_text=rich_text)


class Date(prop.Property):
    def __init__(self, pname, start: datetime = None, end: datetime = None, time_zone: str = None):
        if start is not None:
            start = start.isoformat()
        if end is not None:
            end = end.isoformat()
        kwargs = dict(start=start,
                      end=end,
                      time_zone=time_zone)
        super(Date, self).__init__(pname,
                                   kwargs=kwargs)


class NotionPage:
    def __init__(self, auth=None, page_id=None, parent: Parent = None):
        if auth is None:
            auth = os.environ.get('NOTION_SECRET', None)
            assert auth is not None, 'no auth'

        if page_id is None:
            page_id = os.environ.get('NOTION_PAGE_ID', None)
            if page_id is None:
                assert parent is not None, 'no page id'

        self.req = Request.from_token(authorization=auth)
        self.auth = auth
        if page_id is None:
            assert parent is not None
            res = self.req.post(page_create(), data=Page(parent=parent, properties=Properties()))
            if isinstance(res, Page):
                page_id = res.id
            else:
                assert False, res.to_json(2)

        else:
            res = self.req.get(page_retrieve(page_id))
            assert isinstance(res, Page), res.to_json(2)
        self.page_id = page_id
        print(page_id)
        self.page_object = res
        self.blocks = {}

        self.properties = []
        self.children = []
        self.multi_select_property = defaultdict(set)

    def flush_property(self):
        for property_name, values in self.multi_select_property.items():
            self.properties.append(prop.MultiSelect(property_name, selects=[
                prop.MultiSelectOption(name=v) for v in values
            ]))

        page = Page(properties=Properties(*self.properties))
        res = self.req.patch(
            url=page_update(self.page_id),
            data=page
        )
        self.properties.clear()
        if isinstance(res, NotionError):
            print(res)
        return res

    def flush_children(self):
        page = Page(children=self.children)
        res = self.req.patch(
            url=block_children_append(self.page_id),
            data=page
        )
        self.children.clear()
        if isinstance(res, NotionError):
            print(res)
        return res

    def append_code(self, code, language='python', usage=None):
        code = [code[i:i + 1999] for i in range(0, len(code), 1999)]
        self.children.append(block.Code(rich_text=[rich.Text(f"{i}") for i in code][:100], language=language))

    def set_title(self, property_name, value):
        self.properties.append(prop.Title(property_name,
                                          rich_text=[rich.Text(content=value)]))

    def set_text(self, property_name, value):
        value = f'{value}'
        self.properties.append(RichTextProp(property_name,
                                            rich_text=[rich.Text(content=value)]))

    def set_duration(self, property_name: str, start: datetime = None, end: datetime = None):
        self.properties.append(Date(property_name, start=start, end=end))

    def set_checkbox(self, property_name: str, checkbox=True):
        self.properties.append(prop.CheckBox(property_name, checkbox))

    def set_number(self, property_name: str, number):
        self.properties.append(prop.Number(property_name, number))

    def add_option(self, property_name, value):
        if value is None:
            return
        else:
            value = f'{value}'
        self.multi_select_property[property_name].add(value)
        self.properties.append(prop.MultiSelect(pname=property_name, selects=[
            prop.MultiSelectOption(name=value)
        ]))

    def remove_option(self, property_name, value):
        if value in self.multi_select_property[property_name]:
            self.multi_select_property[property_name].remove(value)

        # return self.flush()


class NotionCallback(callbacks.TrainCallback, callbacks.InitialCallback):
    only_main_process = False

    def ensure_property(self):
        if self.npage is None:
            return
        self.npage

    def on_hooked(self, source: Trainer, params: ParamsType):
        super(NotionCallback, self).on_hooked(source, params)
        self.npage = None
        if params.get('notion_page_id', None) is None:
            source.logger.info('notion_page_id not in params.')
            return
        if not is_main():
            return

        try:
            self.npage = NotionPage(parent=Parent.DataBaseParent(params.notion_page_id))
        except Exception as e:
            source.logger.info(f'{type(e)}{e} happends when create NotionPage.')
            return
        source.exp.dump_info('notion', {
            'type': 'page_id',
            'page_id': self.npage.page_id
        })
        self.npage.set_title('Test Name', source.exp.test_name)
        self.npage.set_text('Experiment Name', source.exp.exp_name)
        self.start = datetime.now().astimezone()
        self.npage.set_duration('Duration', start=self.start)

        source.exp.add_exit_hook(self.finished)

        self.npage.set_text('LogFile', '\n'.join(source.logger.out_channel))

        self.npage.set_text('Params Hash', params.hash())

        self.npage.set_text('GPU number', world_size())

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
        commit_info = trainer.exp.get_prop('git')
        if self.npage is None:
            return
        if 'commit' in commit_info:
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
