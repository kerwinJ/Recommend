from sklearn.cluster import *
from collections import Counter
import numpy as np
import scipy.io as sio

class DatasetLoad:
	def __init__(self, file_name):
		self.file_name = file_name

	def load_data(self):
		data = []
		f = open(self.file_name, 'r')
		lines = f.readlines()
		for line in lines:
			line_temp = line.strip().split(',')
			line_temp = map(int, line_temp)
			data.append(line_temp)
		f.close()
		return data

	def load_data_mat(self):
		dataset = np.zeros((1000, 10000), dtype=np.int16)
		f = open(self.file_name, 'r')
		lines = f.readlines()
		for line in lines:
			line_temp = line.strip().split(',')
			line_temp = map(int, line_temp)
			dataset[line_temp[0]-1, line_temp[1]-1] = 1
		f.close()
		return np.asarray(dataset)

	def load_data_dict(self):
		n_users = 1000
		data_dict = {}
		f = open(self.file_name, 'r')
		lines = f.readlines()
		for i in xrange(n_users):
			data_dict[i] = []
		for line in lines:
			line_temp = line.strip().split(',')
			line_temp = map(int, line_temp)
			data_dict[line_temp[0]-1].append(line_temp[1]-1)
		f.close()
		return data_dict

	def load_data_dict_col(self):
		n_images = 10000
		data_dict = {}
		f = open(self.file_name, 'r')
		lines = f.readlines()
		for i in xrange(n_images):
			data_dict[i] = []
		for line in lines:
			line_temp = line.strip().split(',')
			line_temp = map(int, line_temp)
			data_dict[line_temp[1]-1].append(line_temp[0]-1)
		f.close()
		return data_dict

	def norm_data(self, file_name_feature):
		feat_mat = sio.loadmat(file_name_feature)['featM_vgg']
		n_rows, n_cols = feat_mat.shape
		for i in xrange(n_rows):
			row_sum = np.sum(feat_mat[i])
			for j in xrange(n_cols):
				feat_mat[i,j] /= 10
		sio.savemat('feat_mat_norm', {'feat_mat': feat_mat})

	def load_rows_ground(self):
		file_name_ground = 'data/Q_test1k.mat'
		data = sio.loadmat(file_name_ground)
		rows_ground = data['Q_test1k'][0]
		return rows_ground
if __name__ == '__main__':
	print ' '
