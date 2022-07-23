
from pathlib import Path


def choose_root(*paths) -> Path:
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    raise FileNotFoundError('No appropriate root found.')


CC359_DATA_PATH = choose_root(
    '/home/dsi/shaya/cc359_data/CC359/'
)
MSM_DATA_PATH = choose_root(
    '/home/dsi/shaya/multiSiteMRI/'
)

msm_splits_dir = choose_root('/home/dsi/shaya/data_split_msm2/')
msm_res_dir = choose_root('/home/dsi/shaya/msm_results/')

cc359_res_dir = choose_root('/home/dsi/shaya/spottune_results/')
cc359_splits_dir = choose_root('/home/dsi/shaya/data_splits/')


multiSiteMri_int_to_site = {0:'ISBI',1:"ISBI_1.5",2:'I2CVB',3:"UCL",4:"BIDMC",5:"HK" }