import os
import json
from collections import OrderedDict

from addict import Dict

from src.utils import key_exists
from src.utils.excel_utils import ExcelReader
from src.utils.video_utils import read_video_frames

###############################
### Datasets meta data ########
###############################
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../videos')


class VideoDatasetLoader:
    def __init__(self, dataset_name='manual', split='train',
                 dataset_download_dir=DATASET_DIR):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset_download_dir = dataset_download_dir
        self.dataset_base_dir = os.path.join(dataset_download_dir, f'{dataset_name}/{split}')

        self.dataset_meta_data = None
        self._read_meta_data()

        self.n_episode = 0
        # maintain a video data cache to speed up data reading
        self._video_data_cache = OrderedDict()
        self._max_cache_size = 10

    def _read_meta_data(self):
        meta_data_path = os.path.join(self.dataset_download_dir, 'meta_data.xlsx')
        excel_data_reader = ExcelReader(excel_file_path=meta_data_path,
                                        sheet_name=self.dataset_name.lower())

        n_excel_rows = excel_data_reader.n_rows()
        assert n_excel_rows > 1, 'the excel sheet should have more than 1 rows'
        excel_data = excel_data_reader.get_rows_data(list(range(2, n_excel_rows + 1)),
                                                     append_row_idx=True)
        # read meta data from the excel_data
        # noinspection PyTypeChecker
        self.dataset_meta_data = Dict()
        for i in range(len(excel_data)):
            # converting episode IDs to str can make life easier later
            episode_id = str(int(float(excel_data[i]['episode_id'])))
            self.dataset_meta_data[episode_id] = Dict({
                'episode_id': episode_id,
                'row_idx': excel_data[i]['row_idx'],
                'dataset_name': excel_data[i]['dataset_name'],
                'split': excel_data[i]['split'],
                'instruction': excel_data[i]['instruction'],
                'ground_truth_narration': excel_data[i]['ground_truth_narration'],
                'negative_examples': [e.strip().split('#') for e in excel_data[i]['negative_examples'].split(',')],
                'positive_examples': [e.strip().split('#') for e in excel_data[i]['positive_examples'].split(',')],
                'undesired_behaviors': [b.strip() for b in excel_data[i]['undesired_behaviors'].strip().split('[]')
                                        if b.strip() != ''],
            })
        self.n_episode = len(self.dataset_meta_data)

    def num_episode(self):
        return self.n_episode

    def episode_ids(self):
        return list(self.dataset_meta_data.keys())

    def available_splits(self):
        return [self.split]

    def print_basic_info(self):
        print(f'num of samples in the \"{self.split}\" split: {self.num_episode()} '
              f'| available splits: {self.available_splits()}.')

    def get_video_fname(self, episode_idx):
        return os.path.join(self.dataset_base_dir,
                            f'{self.dataset_name}#split_{self.split}#episode_{episode_idx}#_0.mp4')

    def get_trajectory(self, episode_idx):
        assert isinstance(episode_idx, str), 'episode idx should be given as str'
        assert episode_idx in self.dataset_meta_data, f'episode idx {episode_idx} not in dataset_meta_data'

        if episode_idx not in self._video_data_cache:
            if len(self._video_data_cache) == self._max_cache_size:
                self._video_data_cache.popitem(last=False)
            frames = read_video_frames(self.get_video_fname(episode_idx))
            self._video_data_cache[episode_idx] = frames

        traj_data = Dict(self.dataset_meta_data[episode_idx])
        traj_data.img_obs = self._video_data_cache[episode_idx]
        return traj_data


class CombinedDatasetLoader:
    """
    In the current version of our benchmark, there is only one single dataset and split. But additional datasets
        or splits might be added in the future. This class is for helping maintain multiple datasets & splits.
    """
    def __init__(self):
        self.dataset_loaders = Dict()

    def _get_loader(self, dataset_name, split):
        if not key_exists(self.dataset_loaders, [dataset_name, split])[0]:
            self.dataset_loaders[dataset_name][split] = VideoDatasetLoader(dataset_name, split)
        return self.dataset_loaders[dataset_name][split]

    def num_episode(self, dataset_name, split):
        data_loader = self._get_loader(dataset_name, split)
        return data_loader.num_episode()

    def get_trajectory(self, dataset_name, split, episode_idx):
        data_loader = self._get_loader(dataset_name, split)
        return data_loader.get_trajectory(episode_idx)

    def episode_ids(self, dataset_name, split):
        data_loader = self._get_loader(dataset_name, split)
        return data_loader.episode_ids()
