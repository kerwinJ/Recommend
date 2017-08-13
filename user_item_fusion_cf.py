import numpy as np
import scipy.io as sio
from load_data import DatasetLoad
from item_based_cf import get_image_recommend, cal_sims_mat
from user_based_cf import get_user_recommend
from metrics import cal_precision, cal_recall, cal_f1_score


def get_recommend(user_dict, user_sims, image_sims, topk, w_lambda):
	sims = w_lambda*user_sims + (1.0-w_lambda)*image_sims

	for item in user_dict:
		sims[item] = -1

	sims_list = []
	for i in xrange(len(sims)):
		sims_list.append([i, sims[i]])
	return sorted(sims_list, key=lambda x:x[1], reverse=True)[:topk]

def user_item_based_cf(users_dict, images_dict, topk, rows_ground, w_lambda,user_sims_mat,image_sims_mat):
	n_users = len(users_dict)
	n_images = len(images_dict)
	#user_sims_mat = cal_sims_mat(user_dict)
	#image_sims_mat = cal_sims_mat(image_dict)

	precision = 0.0
	recall = 0.0
	count = 0
	for i in xrange(n_users):
		#if len(users_dict[i])<5:
		#	continue
		count += 1
		user_sims = get_user_recommend(i, users_dict, user_sims_mat)
		image_sims = get_image_recommend(i, users_dict, image_sims_mat)
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

def w_lambda_test(users_dict, images_dict, topk, rows_ground, user_sims_mat,image_sims_mat):
	max_precision = 0.0
	max_lambda = 0.0

	w_lambda = 0.0
	while w_lambda <= 0.2:
		precision, f1_score = user_item_based_cf(users_dict, images_dict, topk, rows_ground, w_lambda,user_sims_mat,image_sims_mat)
		print precision, f1_score
		if precision>max_precision:
			max_precision = precision
			max_lambda = w_lambda
		w_lambda += 0.05
	print max_precision
	print max_lambda

if __name__ == "__main__":
	file_name = "data/UI_1kPos_n0.txt"
	dataset_load = DatasetLoad(file_name)
	users_dict = dataset_load.load_data_dict()
	images_dict = dataset_load.load_data_dict_col()
	rows_ground = dataset_load.load_rows_ground()

	n_users = len(users_dict)
	n_images = len(images_dict)
	user_sims_mat = cal_sims_mat(users_dict)
	image_sims_mat = cal_sims_mat(images_dict)

	topk = 50
	
	
	w_lambda = 0.3 #precision=0.679
	precision, f1_score = user_item_based_cf(users_dict, images_dict, topk, rows_ground, w_lambda,user_sims_mat,image_sims_mat)
	print precision, f1_score
	
	sio.savemat('result/user_item_fusion.mat', {'precision':precision, 'f1_score':f1_score})
	print sio.loadmat('result/user_item_fusion.mat')['precision'][0]
	print sio.loadmat('result/user_item_fusion.mat')['f1_score'][0]
	
	
	
	#w_lambda_test(users_dict, images_dict, topk, rows_ground, user_sims_mat,image_sims_mat)
	
	