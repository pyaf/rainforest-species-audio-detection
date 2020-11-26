import pdb
import yaml
from easydict import EasyDict as edict
from pathlib import Path

def prepare_config(cfg):
    cfg = edict(cfg)
    cfg.batch_size = edict(cfg.batch_size)
    cfg.home = Path(home)
    cfg.data_home = Path(data_home)
    cfg.df_path = Path(df_path)
    cfg.sample_submission = Path(sample_submission)
    cfg.data_folder = Path(data_folder)
    pdb.set_trace()
    return cfg


def load(config_path):
    with open(config_path, 'r') as fid:
        yaml_config = yaml.load(fid)
    config = prepare_config(yaml_config)
    return config
