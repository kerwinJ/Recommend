#from __future__ import division
from load_data import DatasetLoad
from distance import jaccard_dist
from metrics import *
from sklearn.cluster import KMeans
import numpy as np 
import scipy.io as sio


def cal_sims_mat(data_dict):
	n_rows = len(data_dict)
	sims_mat = np.zeros((n_rows, n_rows))
	for i in xrange(n_rows):
		for j in xrange(i+1, n_rows):
			simij = jaccard_dist(data_dict[i], data_dict[j])
			sims_mat[i,j] = simij
			sims_mat[j,i] = simij
	return sims_mat

def get_image_recommend(idx, data_dict, sims_mat):
	n_rows = 10000
	sims = np.zeros(n_rows)
	for i in xrange(n_rows):
		sim_sum = 0.0
		for item in data_dict[idx]:
			sim_sum += sims_mat[i][item]
		sims[i] = sim_sum
	
	return sims

def get_recommend(idx, data_dict, sims_mat, topk):
	
	sims = get_image_recommend(idx, data_dict, sims_mat)
	for item in data_dict[idx]:
		sims[item] = -1

	n_rows = 10000
	sims_list = []
	for i in xrange(n_rows):
		sims_list.append([i, sims[i]])
	return sorted(sims_list, reverse=True, key=lambda x:x[1])[:topk]


def item_based_cf(users_dict, images_dict, image_sims_mat, topk, rows_ground):
	n_users = len(users_dict)

	precision = 0.0
	recall = 0.0
	count = 0
	for i in xrange(n_users):
		#if len(data_dict[i])<5:
		#	continue
		count += 1
		rank = get_recommend(i, users_dict, image_sims_mat, topk)
		
		precision_i = cal_precision(i, rank, rows_ground)
		rat_no = len(users_dict[i])
		recall_i = cal_recall(i, rank, rat_no, rows_ground)
		precision += precision_i
		recall += recall_i
		#print count
	precision /= count
	recall /= count
	f1_score = cal_f1_score(precision, recall)
	return precision, f1_score

if __name__ == "__main__":
	file_name = "data/UI_1kPos_n0.txt"
	dataset_load = DatasetLoad(file_name)
	users_dict = dataset_load.load_data_dict()
	images_dict = dataset_load.load_data_dict_col()
	rows_ground = dataset_load.load_rows_ground()
	image_sims_mat = cal_sims_mat(images_dict)

	topk = 50
	precision, f1_score = item_based_cf(users_dict, images_dict, image_sims_mat, topk, rows_ground)
	sio.savemat('result/item_based.mat', {'precision':precision, 'f1_score':f1_score})
	print sio.loadmat('result/item_based.mat')['precision'][0]
	print sio.loadmat('result/item_based.mat')['f1_score'][0]
	