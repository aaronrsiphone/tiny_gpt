# coding: utf-8
'''
Which forward pass serves a config.json: model_type -> (config, model).
'''

import json
import os

from tinygpt.hf.llama import LlamaConfig, LlamaModel, UnsupportedModel


def _families():
    from tinygpt.hf.lfm3 import Lfm3Config, Lfm3Model
    fam = {t: (LlamaConfig, LlamaModel) for t in LlamaConfig.SUPPORTED_TYPES}
    fam['lfm3'] = (Lfm3Config, Lfm3Model)
    return fam


def supported_types():
    return sorted(_families())


def config_from_dict(d):
    fam = _families().get(d.get('model_type'))
    if fam is None:
        raise UnsupportedModel('model_type %r is not supported (supported: '
                               '%s)' % (d.get('model_type'),
                                        ', '.join(supported_types())))
    return fam[0](d)


def read_config(path):
    with open(os.path.join(path, 'config.json'), 'rb') as f:
        return json.loads(f.read().decode('utf-8'))


def model_from_dir(path, **kw):
    '''Load the model in a snapshot directory with the right class.'''
    d = read_config(path)
    config_from_dict(d)   # raises UnsupportedModel with the full list
    return _families()[d['model_type']][1].from_dir(path, **kw)
