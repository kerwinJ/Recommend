# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 16:12:55 2017

@author: jacky
"""
from sklearn.cluster import *
from collections import Counter
from load_data import DatasetLoad
from distance import jaccard_dist
from metrics import *
import numpy as np 
import scipy.io as sio
import pickle

def init_centers(data_dict, n_class):
	'''
	centers = {}
	n_rows = len(data_dict)
	centers_ids = np.asarray(np.random.randint(0, n_rows, n_class))

	for k in xrange(n_class):
		centers[k] = data_dict[centers_ids[k]]
	with open('kmean_init.pickle', 'wb') as f:
		pickle.dump(centers, f)
	'''
	with open('kmean_init.pickle', 'rb') as f:
		centers = pickle.load(f)
	return centers

def update_centers(data_dict, rows_class, n_class):
	centers = {}
	for k in xrange(n_class):
		centers[k] = []
	for k in xrange(n_class):
		center_vec = np.zeros(10000)
		for user in rows_class[k]:
			items = data_dict[user]
			center_vec[items] += 1
		for j in xrange(len(center_vec)):
			if center_vec[j] > 0.08*len(rows_class[k]):
				centers[k].append(j)
	return centers


def kmeans(data_dict, n_class):
	n_users = len(data_dict)
	centers = init_centers(data_dict, n_class)
	cluster_changed = True
	rows_class = {}
	for k in xrange(n_class):
		rows_class[k] = []
	count = 0
	while(cluster_changed & (count<30)):
		count += 1
		print count
		cluster_changed = False
		rows_class_temp = {}
		for k in xrange(n_class):
			rows_class_temp[k] = []
		for i in xrange(n_users):
			cluster_dist = np.zeros(n_class)
			for k in xrange(n_class):
				cluster_dist[k] = jaccard_dist(data_dict[i], centers[k])
			cls_idx = np.argmax(cluster_dist)
			rows_class_temp[cls_idx].append(i)
		for k in xrange(n_class):
			if rows_class[k] != rows_class_temp[k]:
				cluster_changed = True
			rows_class[k] = rows_class_temp[k]
		centers = update_centers(data_dict, rows_class, n_class)
	return rows_class, centers

def get_cluster_no(user, centers):
	n_class = len(centers)
	sims = np.zeros(n_class)
	for i in xrange(n_class):
		sims[i] = jaccard_dist(user, centers[i])
	sims_list = []
	for i in xrange(len(sims)):
		sims_list.append([i, sims[i]])
	return sorted(sims_list, reverse=True, key=lambda x:x[1])[:2]

def get_image_sims(idx, data_dict, n_images, cluster_k):
	user_sims = []
	for i in xrange(len(cluster_k)):
		user_sims.append([cluster_k[i], jaccard_dist(data_dict[idx], data_dict[cluster_k[i]])])
	
	image_sims = np.zeros(n_images)
	for i in xrange(len(user_sims)):
		user_no = user_sims[i][0]
		user_sim = user_sims[i][1]
		for item in data_dict[user_no]:
			image_sims[item] += user_sim
	return image_sims

def get_sims_smooth(data_dict, n_images, cluster_k):
	sims_smooth = np.zeros(n_images)
	sims = np.zeros(n_images)
	for i in xrange(len(cluster_k)):
		for item in data_dict[cluster_k[i]]:
			sims_smooth[item] += 1
	for i in xrange(n_images):
		if sims_smooth[i] > 0.1*len(cluster_k):
			sims[i] = 1
	return sims

def get_user_recommend(idx, data_dict, rows_class, centers):
	n_images = 10000
	n_class = len(centers)
	cluster_list = get_cluster_no(data_dict[idx], centers)

	sims = np.zeros(n_images)
	sims_smooth = np.zeros(n_images)
	for k in xrange(len(cluster_list)):
		cluster_no = cluster_list[k][0]
		cluster_sim = cluster_list[k][1]
		sims_smooth += cluster_sim * get_sims_smooth(data_dict, n_images, rows_class[cluster_no])
		sims += cluster_sim * get_image_sims(idx, data_dict, n_images, rows_class[cluster_no])
	w_smooth = 0.01
	#sims = w_smooth*sims_smooth + (1-w_smooth)*sims

	return sims

def get_recommend(idx, data_dict, rows_class, centers, topk):
	n_images = 10000

	sims = get_user_recommend(idx, data_dict, rows_class, centers)
	for item in data_dict[idx]:
		sims[item] = -1

	sims_list = []
	for i in xrange(n_images):
		sims_list.append([i, sims[i]])
	return sorted(sims_list, reverse=True, key=lambda x:x[1])[:topk]

def user_cluster_based_cf(data_dict, rows_ground, rows_class, centers, topk):
	n_users = len(data_dict)

	precision = 0.0
	recall = 0.0
	count = 0
	for i in xrange(n_users):
		#if len(data_dict[i]) < 5:
		#	continue
		count += 1
		rank = get_recommend(i, data_dict, rows_class, centers, topk)

		precision_i = cal_precision(i, rank, rows_ground)
		rat_no = len(data_dict[i])
		recall_i = cal_recall(i, rank, rat_no, rows_ground)
		precision += precision_i
		recall += recall_i
		#print count
	precision /= count
	recall /= count
	f1_score = cal_f1_score(precision, recall)
	return precision, f1_score

def n_class_test(users_dict, rows_ground, rows_class, centers, topk):
	max_precision = 0.0
	max_nclass = 0
	n_class = 100

	while(n_class<=200):
		rows_class, centers = kmeans(users_dict, n_class)
		precision, recall = user_cluster_based_cf(users_dict, rows_ground, rows_class, centers, topk)
		
		if precision > max_precision:
			max_precision = precision
			max_nclass = n_class
		n_class += 10
	print max_nclass
	print max_precision

if __name__ == "__main__":

	file_name = "data/UI_1kPos_n0.txt"
	dataset_load = DatasetLoad(file_name)
	users_dict = dataset_load.load_data_dict()
	images_dict = dataset_load.load_data_dict_col()
	rows_ground = dataset_load.load_rows_ground()

	n_class = 200 #109
	rows_class, centers = kmeans(users_dict, n_class)

	topk = 50
	#n_class_test(users_dict, rows_ground, rows_class, centers, topk)
	
	precision, f1_score = user_cluster_based_cf(users_dict, rows_ground, rows_class, centers, topk)
	print precision, f1_score
	
	sio.savemat('result/user_cluster_based.mat', {'precision':precision, 'f1_score':f1_score})
	print sio.loadmat('result/user_cluster_based.mat')['precision'][0]
	print sio.loadmat('result/user_cluster_based.mat')['f1_score'][0]
	
	