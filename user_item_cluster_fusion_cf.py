import numpy as np
import scipy.io as sio
from load_data import DatasetLoad
from user_cluster_cf import kmeans, get_user_recommend
from item_cluster_cf_multi import get_image_recommend
from metrics import *

def get_recommend(user_dict, user_sims, image_sims, topk, w_lambda):
	sims = w_lambda*user_sims + (1.0-w_lambda)*image_sims
	for item in user_dict:
		sims[item] = -1

	sims_list = []
	for i in xrange(len(sims)):
		sims_list.append([i, sims[i]])
	return sorted(sims_list, key=lambda x:x[1], reverse=True)[:topk]

def user_item_fusion_cf_multi(w_lambda, data_mat, users_dict, images_dict, users_class, users_centers, images_class, topk, rows_ground):
	n_users, n_images = data_mat.shape

	precision = 0.0
	recall = 0.0
	count = 0
	for i in xrange(n_users):
		#if len(users_dict[i])<5:
		#	continue
		count += 1
		user_sims = get_user_recommend(i, users_dict, users_class, users_centers)
		image_sims = get_image_recommend(i, data_mat, users_dict, images_dict, images_class)
		rank = get_recommend(users_dict[i], user_sims, image_sims, topk, w_lambda)

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

def get_rows_class(G_mat):
	n_images, n_class = G_mat.shape
	rows_class = {}
	for k in xrange(n_class):
		rows_class[k] = []
		for i in xrange(n_images):
			if G_mat[i, k] == 1:
				rows_class[k].append(i)
	return rows_class

def w_lambda_test(w_lambda, data_mat, users_dict, images_dict, users_class, users_centers, images_class, topk, rows_ground):
	
	w_lambda = 0.0

	max_precision = 0.0
	while w_lambda <=0.5:
		precision, f1_score = user_item_fusion_cf_multi(w_lambda, data_mat, users_dict, images_dict, users_class, users_centers, images_class, topk, rows_ground)
		print precision, f1_score
		if max_precision<precision:
			max_precision = precision
			max_lambda= w_lambda
		w_lambda += 0.05
	print max_precision
	print max_lambda

if __name__ == "__main__":
	

	file_name = "data/UI_1kPos_n0.txt"
	dataset_load = DatasetLoad(file_name)
	data_mat = dataset_load.load_data_mat()
	users_dict = dataset_load.load_data_dict()
	images_dict = dataset_load.load_data_dict_col()
	rows_ground = dataset_load.load_rows_ground()

	G_mat = sio.loadmat('multi-view/G_mat.mat')['G_mat']
	F_mat = sio.loadmat('multi-view/F_mat.mat')['F_mat']
	images_class = get_rows_class(G_mat)
	n_class = 130
	users_class, users_centers = kmeans(users_dict, n_class)

	topk = 50
	w_lambda = 0.05
	
	precision, f1_score = user_item_fusion_cf_multi(w_lambda, data_mat, users_dict, images_dict, users_class, users_centers, images_class, topk, rows_ground)
	sio.savemat('result/user_item_fusion_multi.mat', {'precision':precision, 'f1_score':f1_score})
	print sio.loadmat('result/user_item_fusion_multi.mat')['precision'][0]
	print sio.loadmat('result/user_item_fusion_multi.mat')['f1_score'][0]
	
	#w_lambda_test(w_lambda, data_mat, users_dict, images_dict, users_class, users_centers, images_class, topk, rows_ground)