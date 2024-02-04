from torch.utils.data import Dataset
from dataloaders.loader_utils import get_final_data
import torch

class YearOpenSplitDataSet(Dataset):

	def __init__(self, slam_years, get_match_info=False):
		self.data = get_final_data(slam_years, get_match_info=get_match_info)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		return self.data[idx]