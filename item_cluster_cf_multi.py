import numpy as np 
import scipy.io as sio
from load_data import DatasetLoad
from distance import jaccard_dist
from metrics import *

def get_rows_class(G_mat):
	n_images, n_class = G_mat.shape
	rows_class = {}
	for k in xrange(n_class):
		rows_class[k] = []
		for i in xrange(n_images):
			if G_mat[i, k] == 1:
				rows_class[k].append(i)
	return rows_class

def get_sims_smooth(data_dict, n_images, cluster_k):
	sims_smooth = np.zeros(n_images)
	sims = np.zeros(n_images)
	for i in xrange(len(cluster_k)):
		for item in data_dict[cluster_k[i]]:
			sims_smooth[item] += 1
	for i in xrange(n_images):
		if sims_smooth[i] > 0.3*len(cluster_k):
			sims[i] = 1
	return sims

def get_image_recommend(idx, data_mat, users_dict, images_dict, rows_class):
	n_users, n_images = data_mat.shape
	n_class = len(rows_class)
	cluster_sims_list = []
	for k in xrange(n_class):
		cluster_sims_list.append([k, jaccard_dist(users_dict[idx], rows_class[k])])
	cluster_list = sorted(cluster_sims_list, reverse=True, key=lambda x: x[1])[:2]

	sims = np.zeros(n_images)
	sims_smooth = np.zeros(n_images)
	for k in xrange(len(cluster_list)):
		cluster_no = cluster_list[k][0]
		cluster_sim = cluster_list[k][1]
		for item in rows_class[cluster_no]:
			for j in users_dict[idx]:
				sims[item] += cluster_sim*jaccard_dist(images_dict[item], images_dict[j])
		sims_smooth = cluster_sim*get_sims_smooth(images_dict, n_images, rows_class[cluster_no])

	w_smooth = 0.01
	#sims = w_smooth*sims_smooth + (1-w_smooth)*sims
	return sims


def get_recommend(idx, data_mat, users_dict, images_dict, rows_class, topk):
	n_users, n_images = data_mat.shape
	sims = get_image_recommend(idx, data_mat, users_dict, images_dict, rows_class)
	for item in users_dict[idx]:
		sims[item] = -1

	sims_list = []
	for i in xrange(n_images):
		sims_list.append([i, sims[i]])
	return sorted(sims_list, reverse=True, key=lambda x:x[1])[:topk]


def cluster_based_multi_cf(data_mat, users_dict, images_dict, rows_class, topk, rows_ground):
	n_users, n_images = data_mat.shape

	precision = 0.0
	recall = 0.0
	count = 0
	for i in xrange(n_users):
		#if len(users_dict[i])<5:
		#	continue
		count += 1
		rank = get_recommend(i, data_mat, users_dict, images_dict, rows_class, topk)

		precision_i = cal_precision(i, rank, rows_ground)
		rat_no = len(users_dict[i])
		recall_i = cal_recall(i, rank, rat_no, rows_ground)
		precision += precision_i
		recall += recall_i
	precision /= count
	recall /= count
	f1_score = cal_f1_score(precision, recall)
	return precision, f1_score

if __name__ == '__main__':

	G_mat = sio.loadmat('multi-view/G_mat.mat')['G_mat']
	F_mat = sio.loadmat('multi-view/F_mat.mat')['F_mat']
	rows_class = get_rows_class(G_mat)

	file_name = "data/UI_1kPos_n0.txt"
	dataset_load = DatasetLoad(file_name)
	data_mat = dataset_load.load_data_mat()
	users_dict = dataset_load.load_data_dict()
	images_dict = dataset_load.load_data_dict_col()
	rows_ground = dataset_load.load_rows_ground()
	
	topk = 50
	precision, f1_score = cluster_based_multi_cf(data_mat, users_dict, images_dict, rows_class, topk, rows_ground)
	print precision, f1_score
	
	sio.savemat('result/cluster_based_multi_48.mat', {'precision':precision, 'f1_score':f1_score})
	print sio.loadmat('result/cluster_based_multi_48.mat')['precision'][0]
	print sio.loadmat('result/cluster_based_multi_48.mat')['f1_score'][0]
	