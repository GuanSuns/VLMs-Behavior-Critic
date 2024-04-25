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

        self.episode_ids = set()
        self._test_cases = list()
        self._read_meta_data()

        # maintain a video data cache to speed up data reading
        self._video_data_cache = OrderedDict()
        self._max_cache_size = 200

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
        self.episode_ids, self._test_cases = set(), list()
        for i in range(len(excel_data)):
            # converting episode IDs to str can make life easier later
            episode_id = str(int(float(excel_data[i]['episode_id'])))

            test_case = Dict({
                'episode_id': episode_id,
                'row_idx': excel_data[i]['row_idx'],
                'dataset_name': excel_data[i]['dataset_name'],
                'split': excel_data[i]['split'],
                'instruction': excel_data[i]['instruction'],
                'ground_truth_narration': excel_data[i]['ground_truth_narration'],
                'negative_examples': [e.strip().split('#')[-1] for e in excel_data[i]['negative_examples'].split(',')],
                'positive_examples': [e.strip().split('#')[-1] for e in excel_data[i]['positive_examples'].split(',')],
                'undesired_behaviors': [b.strip() for b in excel_data[i]['undesired_behaviors'].strip().split('[]')
                                        if b.strip() != '']
            })
            self._test_cases.append(test_case)
            self.episode_ids.update([episode_id] + test_case['negative_examples'] + test_case['positive_examples'])

    @property
    def test_cases(self):
        return self._test_cases

    def num_test_cases(self):
        return len(self._test_cases) if isinstance(self._test_cases, list) else -1

    def available_splits(self):
        return [self.split]

    def print_basic_info(self):
        print(f'num of test cases in the \"{self.split}\" split: {self.num_test_cases()} '
              f'| available splits: {self.available_splits()}.')

    def get_video_fname(self, episode_id):
        return os.path.join(self.dataset_base_dir,
                            f'{self.dataset_name}#split_{self.split}#episode_{episode_id}#_0.mp4')

    def get_traj_data(self, episode_id):
        assert isinstance(episode_id, str), 'episode id should be given as str'
        assert episode_id in self.episode_ids, f'episode id {episode_id} not in dataset_meta_data'

        if episode_id not in self._video_data_cache:
            if len(self._video_data_cache) == self._max_cache_size:
                self._video_data_cache.popitem(last=False)
            frames = read_video_frames(self.get_video_fname(episode_id))
            self._video_data_cache[episode_id] = frames

        traj_data = Dict({'img_obs': self._video_data_cache[episode_id]})
        return traj_data
