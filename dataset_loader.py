import glob
import pickle
import torch
from torch.utils import data
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from models.preprocessing import *
import random

# Code for this dataloader is heavily borrowed from PECNet.
# https://github.com/HarshayuGirase/Human-Path-Prediction


def initial_pos_func(traj_batches):
	batches = []
	for b in traj_batches:
		starting_pos = b[:, 7, :].copy()/1000 #starting pos is end of past, start of future. scaled down.
		batches.append(starting_pos)

	return batches


class SocialDataset(data.Dataset):

	def __init__(self, folder, scene, set_name="train", b_size=4096, t_tresh=60, d_tresh=50, homo_matrix=None, verbose=False):
		'Initialization'
		load_name = "/home/tyl/code/930idea/{0}/social_{1}_{2}_{3}_{4}_{5}.pickle".format(folder, scene, set_name, b_size, t_tresh, d_tresh)

		# print(load_name)
		with open(load_name, 'rb') as f:
			data = pickle.load(f)

		traj, masks = data
		traj_new = []

		for t in traj:
			t = np.array(t)
			t = t[:, :, 2:4]
			traj_new.append(t)
			if set_name == "train":
				#augment training set with reversed tracklets...
				# reverse_t = np.flip(t, axis=1).copy()
				ks = [1, 2, 3]
				for k in ks:
					data_rot = rot(t, k).copy()
					traj_new.append(data_rot)
				data_flip = fliplr(t).copy()
				traj_new.append(data_flip)

		masks_new = []
		for m in masks:
			masks_new.append(m)

			if set_name == "train":
				#add second time for the reversed tracklets...
				masks_new.append(m)
				for _ in range(3):
					masks_new.append(m)

		seq_start_end_list = []
		for m in masks:
			total_num = m.shape[0]
			scene_start_idx = 0
			num_list = []
			for i in range(total_num):
				if i < scene_start_idx:
					continue
				scene_actor_num = np.sum(m[i])
				scene_start_idx += scene_actor_num
				num_list.append(scene_actor_num)

			cum_start_idx = [0] + np.cumsum(np.array(num_list)).tolist()
			seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
			seq_start_end_list.append(seq_start_end)
			if set_name == "train":
				#add second time for the reversed tracklets...
				seq_start_end_list.append(seq_start_end)
				for _ in range(3):
					seq_start_end_list.append(seq_start_end)

		# print(len(traj_new), len(seq_start_end_list), len(masks_new))
		# traj_new = np.array(traj_new, dtype=object)
		# traj_new = np.array(traj_new)
		# masks_new = np.array(masks_new)

		self.trajectory_batches = traj_new.copy()
		self.mask_batches = masks_new.copy()
		# self.initial_pos_batches = np.array(initial_pos_func(self.trajectory_batches), dtype=object) #for relative positioning
		self.initial_pos_batches = initial_pos_func(self.trajectory_batches) #for relative positioning
		self.seq_start_end_batches = seq_start_end_list
		if verbose:
			print("Initialized social dataloader...")

	def __len__(self):
		return len(self.trajectory_batches)

	def __getitem__(self, idx):
		trajectory = self.trajectory_batches[idx]
		mask = self.mask_batches[idx]
		initial_pos = self.initial_pos_batches[idx]
		seq_start_end = self.seq_start_end_batches[idx]
		return np.array(trajectory), np.array(mask), np.array(initial_pos), np.array(seq_start_end)


def socialtraj_collate(batch):
	trajectories = []
	mask = []
	initial_pos = []
	seq_start_end = []
	for _batch in batch:
		trajectories.append(_batch[0])
		mask.append(_batch[1])
		initial_pos.append(_batch[2])
		seq_start_end.append(_batch[3])

	trajectories = np.array(trajectories)
	mask = np.array(mask)
	initial_pos = np.array(initial_pos)
	seq_start_end = np.array(seq_start_end)
	return torch.Tensor(trajectories).squeeze(0), torch.Tensor(mask).squeeze(0), torch.Tensor(initial_pos).squeeze(0), torch.tensor(seq_start_end ,dtype=torch.int32).squeeze(0)


class TrajectoryDataset(data.Dataset):
    def __init__(
        self,
        dataset_path,
        dataset_name,
        dataset_type,
        translation=False,
        rotation=False,
        scaling=False,
        obs_len=8,
        max_neis_num=50,
        dist_threshold=2,
        smooth=False,
        use_augmentation=True,
    ):
        self.translation = translation  # True
        self.rotation = rotation  # True
        self.obs_len = obs_len  # 8
        self.scaling = scaling  # True
        self.max_neis_num = max_neis_num  # 50
        self.dist_threshold = dist_threshold  # 5
        self.smooth = smooth  # False
        self.window_size = 3

        # <_io.BufferedRandom name='./dataset/sdd_train.pkl'>
        f = open(dataset_path + dataset_name + "_" + dataset_type + ".pkl", "rb+")
        self.scenario_list = pickle.load(f)
        print(f"The length of scenario_list is {len(self.scenario_list)}")
        f.close()

        # 模仿PPT扩充数据集
        if use_augmentation:
            raw_len = len(self.scenario_list)
            for i in range(raw_len):
                traj = self.scenario_list[i]

                # 旋转
                ks = [1, 2, 3]
                for k in ks:
                    traj_rot = []
                    traj_rot.append(self.rot(traj[0][:, :2], k))
                    traj_rot.append(self.rot(traj[1], k))
                    traj_rot.append(self.rot(traj[2][:, :, :2], k))
                    self.scenario_list.append(tuple(traj_rot))

                # 水平翻转
                traj_flip = []
                traj_flip.append(self.fliplr(traj[0][:, :2]))
                traj_flip.append(self.fliplr(traj[1]))
                traj_flip.append(self.fliplr(traj[2][:, :, :2]))
                self.scenario_list.append(tuple(traj_flip))

    @staticmethod
    def rot(data, k=1):
        """
        Rotates image and coordinates counter-clockwise by k * 90° within image origin
        :param df: Pandas DataFrame with at least columns 'x' and 'y'
        :param image: PIL Image
        :param k: Number of times to rotate by 90°
        :return: Rotated Dataframe and image
        """
        data_ = data.copy()

        c, s = np.cos(-k * np.pi / 2), np.sin(-k * np.pi / 2)
        R = np.array([[c, s], [-s, c]])  # 旋转矩阵
        data_ = np.dot(data_, R)  # 旋转数据
        return data_

    @staticmethod
    def fliplr(data):
        """
        Flip image and coordinates horizontally
        :param df: Pandas DataFrame with at least columns 'x' and 'y'
        :param image: PIL Image
        :return: Flipped Dataframe and image
        """
        data_ = data.copy()
        R = np.array([[-1, 0], [0, 1]])
        data_ = np.dot(data_, R)

        return data_

    def coll_fn(self, scenario_list):
        # batch <list> [[ped, neis]]]
        ped, neis = [], []
        shift = []

        n_neighbors = []

        for item in scenario_list:
            ped_obs_traj, ped_pred_traj, neis_traj = (
                item[0],
                item[1],
                item[2],
            )  # [T 2] [N T 2] N is not a fixed number  取出来观察帧,预测帧,邻居轨迹

            # 拼接轨迹
            ped_traj = np.concatenate(
                (ped_obs_traj[:, :2], ped_pred_traj), axis=0
            )  # (20, 2) 拼接完整的行人轨迹
            neis_traj = neis_traj[:, :, :2].transpose(
                1, 0, 2
            )  # (N, 20, 2) 邻居轨迹  N表示邻居数量 可能为0
            neis_traj = np.concatenate(
                (np.expand_dims(ped_traj, axis=0), neis_traj), axis=0
            )  # (1+N, 20, 2)  行人和邻居轨迹

            # 计算行人和邻居之间的距离
            distance = np.linalg.norm(
                np.expand_dims(ped_traj, axis=0) - neis_traj, axis=-1
            )  # (1+N, 20)  计算行人和邻居之间的距离
            distance = distance[:, : self.obs_len]  # 取出来前八帧的距离  (1+N, 8)
            distance = np.mean(distance, axis=-1)  # mean distance  取出来和每个邻居的观察帧的平均距离  (1+N, )
            # distance = distance[:, -1] # final distance
            neis_traj = neis_traj[distance < self.dist_threshold]  # 取出来距离小于阈值的邻居轨迹

            n_neighbors.append(neis_traj.shape[0])  # 邻居数目,若只有1个邻居,则表示该行人本身

            origin = ped_traj[self.obs_len - 1 : self.obs_len]  # 取出来行人的第八帧观察帧数据  (1, 2)
            ped_traj = ped_traj - origin  # 当前行人减去观察帧数据
            if neis_traj.shape[0] != 0:
                neis_traj = neis_traj - np.expand_dims(origin, axis=0)  # 邻居减去当前行人观察帧数据

            shift.append(origin)

            if self.rotation:  # 旋转数据
                angle = random.random() * np.pi  # 随机旋转的角度
                rot_mat = np.array(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )
                ped_traj = np.matmul(ped_traj, rot_mat)  # 旋转行人轨迹
                if neis_traj.shape[0] != 0:
                    rot_mat = np.expand_dims(rot_mat, axis=0)
                    rot_mat = np.repeat(rot_mat, neis_traj.shape[0], axis=0)
                    neis_traj = np.matmul(neis_traj, rot_mat)  # 旋转邻居轨迹

            if self.smooth:  # False
                pred_traj = ped_traj[self.obs_len :]
                x_len = pred_traj.shape[0]
                x_list = []
                keep_num = int(np.floor(self.window_size / 2))
                for i in range(self.window_size):
                    x_list.append(pred_traj[i : x_len - self.window_size + 1 + i])
                x = sum(x_list) / self.window_size
                x = np.concatenate(
                    (pred_traj[:keep_num], x, pred_traj[-keep_num:]), axis=0
                )
                ped_traj = np.concatenate((ped_traj[: self.obs_len], x), axis=0)

            # if self.scaling:
            #     scale = np.random.randn(ped_traj.shape[0])*0.05+1
            #     scale = scale.reshape(ped_traj.shape[0], 1)
            #     ped_traj = ped_traj * scale
            #     if neis_traj.shape[0] != 0:
            #         neis_traj = neis_traj * scale

            ped.append(ped_traj)
            neis.append(neis_traj)

        max_neighbors = max(n_neighbors)  # 当前batch最大邻居数目
        neis_pad = []
        neis_mask = []
        for neighbor, n in zip(neis, n_neighbors):  # 遍历每个行人的邻居
            neis_pad.append(
                np.pad(neighbor, ((0, max_neighbors - n), (0, 0), (0, 0)), "constant")
            )  # 邻居轨迹填充成相同的长度
            mask = np.zeros((max_neighbors, max_neighbors))
            mask[:n, :n] = 1  # mask表示是否有邻居, 若为0表示没有邻居, 是填充值
            neis_mask.append(mask)

        ped = np.stack(ped, axis=0)  # (512, 20, 2)  512表示batch_size
        neis = np.stack(
            neis_pad, axis=0
        )  # (512, 1, 20, 2)  邻居的轨迹  1表示当前batch的行人最多只有一个邻居
        neis_mask = np.stack(neis_mask, axis=0)  # (512, 1, 1)  mask表示是否有邻居
        shift = np.stack(shift, axis=0)  # (512, 1, 2)  第八帧数据

        ped = torch.tensor(ped, dtype=torch.float32)
        neis = torch.tensor(neis, dtype=torch.float32)
        neis_mask = torch.tensor(neis_mask, dtype=torch.int32)
        shift = torch.tensor(shift, dtype=torch.float32)  # 第八帧数据
        return ped, neis, neis_mask, shift

    def __len__(self):
        return len(self.scenario_list)

    def __getitem__(self, item):
        return self.scenario_list[item]
