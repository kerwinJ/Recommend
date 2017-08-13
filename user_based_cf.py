# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 16:12:55 2017

@author: jacky
"""
import numpy as np
import scipy.io as sio
from load_data import DatasetLoad
from metrics import *
from distance import jaccard_dist

def cal_sims_mat(data_dict):
	n_rows = len(data_dict)
	sims_mat = np.zeros((n_rows, n_rows))
	for i in xrange(n_rows):
		for j in xrange(i+1, n_rows):
			simij = jaccard_dist(data_dict[i], data_dict[j])
			sims_mat[i,j] = simij
			sims_mat[j,i] = simij
	return sims_mat

def get_sim_users(row_dict, sims_vec):
	sim_users_list = []
	for i in xrange(len(sims_vec)):
		sim_users_list.append([i, sims_vec[i]])
	sim_users = sorted(sim_users_list, reverse=True, key=lambda x:x[1])[:10]
	return sim_users

def get_user_recommend(idx, data_dict, sims_mat):

	sims_users = get_sim_users(data_dict[idx], sims_mat[idx])
	
	sims = np.zeros(10000)
	for i in xrange(len(sims_users)):
		user_no = sims_users[i][0]
		user_sim = sims_users[i][1]
		
		for item in data_dict[user_no]:
			sims[item] += 1*user_sim
	
	return sims

def get_recommend(idx, data_dict, sims_mat, topk):

	sims_users = get_sim_users(data_dict[idx], sims_mat[idx])
	sims = get_user_recommend(idx, data_dict, sims_mat)
	for item in data_dict[idx]:
		sims[item] = -1

	n_images = 10000
	sims_list = []
	for i in xrange(n_images):
		sims_list.append([i, sims[i]])
	return sorted(sims_list, key=lambda x:x[1], reverse=True)[0:topk]
	'''
	sims = np.zeros(10000)
	for i in xrange(len(sims_users)):
		user_no = sims_users[i][0]
		user_sim = sims_users[i][1]
		
		for item in data_dict[user_no]:
			sims[item] += 1*user_sim
	sims_list = []
	for i in xrange(len(sims)):
		sims_list.append([i, sims[i]])
	rank = sorted(sims_list, reverse=True, key=lambda x: x[1])[:k]
	return rank
	'''

def user_based_cf(data_dict, user_sims_mat, topk, rows_ground):
	n_users = len(data_dict)

	precision = 0.0
	recall = 0.0
	count = 0
	for i in xrange(n_users):
		#if len(data_dict[i])<5:
		#	continue
		count += 1
		rank = get_recommend(i, data_dict, user_sims_mat, topk)
		
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

if __name__ == '__main__':
	file_name = 'data/UI_1kPos_n0.txt'
	dataset_load = DatasetLoad(file_name)
	users_dict = dataset_load.load_data_dict()
	rows_ground = dataset_load.load_rows_ground()
	user_sims_mat = cal_sims_mat(users_dict)
	
	topk = 50

	precision, f1_score = user_based_cf(users_dict,user_sims_mat, topk, rows_ground)
	#print precision, f1_score
	
	sio.savemat('result/user_based.mat', {'precision':precision, 'f1_score':f1_score})
	print sio.loadmat('result/user_based.mat')['precision'][0]
	print sio.loadmat('result/user_based.mat')['f1_score'][0]
	

	

