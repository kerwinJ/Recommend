from collections import Counter
from load_data import DatasetLoad
from distance import jaccard_dist
import numpy as np 
import scipy.io as sio

def load_multi_data(data_mat):
	X_mats = []
	X_mats.append(data_mat)
	X_mats.append(sio.loadmat("multi-view/featM_vgg.mat")['featM_vgg'].T)
	return X_mats

def initialize_G(n_rows, n_class):
	'''
	G_mat = np.zeros((n_rows, n_class))
	for i in xrange(n_rows):
		k_col = np.random.randint(n_class)
		G_mat[i][col] = 1
	sio.savemat('multi-view/init_Gmat', {'init_Gmat':G_mat})
	'''
	G_mat = sio.loadmat('multi-view/init_Gmat')['init_Gmat'][:,:n_class]
	return G_mat

def calculate_D(D_mats, w_vec, exp_param):
	n_views = len(D_mats)
	for v in xrange(n_views):
		D_mats[v] = np.power(w_vec[v], exp_param)*D_mats[v]
	return D_mats

def update_F(X_mat, D_mat, G_mat):
	temp_mat = np.dot(D_mat, G_mat)
	try:
		inv_mat = np.linalg.inv(np.dot(G_mat.T, temp_mat))
	except Exception as e:
		print np.dot(G_mat.T, temp_mat)
		
	F_mat = np.dot(np.dot(X_mat, temp_mat), inv_mat)
	return F_mat

def update_g(idx, X_mats, D_mats, F_mats, n_class):
	n_views = len(X_mats)
	residue_vec = np.zeros(n_class)

	for k in xrange(n_class):
		for v in xrange(n_views):
			temp_vec = X_mats[v][:, idx] - F_mats[v][:, k]
			residue_vec[k] += D_mats[v][idx, idx]*np.sum(temp_vec**2)
	cls_idx = np.argmin(residue_vec)
	g_vec = np.zeros(n_class)
	g_vec[cls_idx] = 1
	return g_vec

def calculate_E(X_mat, G_mat, F_mats):
	return X_mat.T - np.dot(G_mat, F_mats.T)

def update_D(D_mat, E_mat):
	for i in xrange(len(D_mat)):
		D_mat[i, i] = 1/(2*np.sum(E_mat[i]**2))
	return 0

def update_w(w_vec, E_mats, D_mats, exp_param):
	n_views = len(E_mats)
	H_traces = np.zeros(n_views)
	for v in xrange(n_views):
		temp_mat = np.dot(E_mats[v].T, D_mats[v])
		temp_mat = np.dot(temp_mat, E_mats[v])
		H_traces[v] = np.trace(temp_mat)
		w_vec[v] = np.power(exp_param*H_traces[v], 1/float(1-exp_param))
	for v in xrange(n_views):
		w_vec[v] = w_vec[v]/np.sum(w_vec)
	return w_vec

def multi_view_kmeans(X_mats, n_images, n_class, exp_param):
	'''
	X_mats
	G_mat
	F_mats
	'''
	n_views = len(X_mats)
	G_mat = initialize_G(n_images, n_class)
	D_mats = []
	for v in xrange(n_views):
		D_mats.append(np.eye(n_images))
	w_vec = []
	for v in xrange(n_views):
		w_vec.append(1.0/n_views)
	iters = 0
	while(iters<7):
		print iters
		calculate_D(D_mats, w_vec, exp_param)

		F_mats = []
		for v in xrange(n_views):
			F_mats.append(update_F(X_mats[v], D_mats[v], G_mat))

		for i in xrange(n_images):
			G_mat[i] = update_g(i, X_mats, D_mats, F_mats, n_class)

		E_mats = []
		for v in xrange(n_views):
			E_mats.append(calculate_E(X_mats[v], G_mat, F_mats[v]))

		for v in xrange(n_views):
			update_D(D_mats[v], E_mats[v])

		update_w(w_vec, E_mats, D_mats, exp_param)
		iters += 1
	sio.savemat('multi-view/G_mat.mat', {'G_mat':G_mat})
	sio.savemat('multi-view/F_mat.mat', {'F_mat':F_mats[0]})

def test_cluster_accuracy():
	G_mat = sio.loadmat('multi-view/G_mat.mat')['G_mat']
	n_images, n_class = G_mat.shape
	rows_class = {}
	for k in xrange(n_class):
		rows_class[k] = []
		for i in xrange(n_images):
			if G_mat[i,k] == 1:
				rows_class[k].append(i//100)
	accuracy = 0.0
	for k in xrange(n_class):
		most_common = Counter(rows_class[k]).most_common(1)
		right_num = float(most_common[0][1])
		accuracy += right_num/len(rows_class[k])
	return accuracy/n_class

def vilidation_test(X_mats, n_images, n_class, exp_param):
	exp_param = 2

	max_purity = 0.0
	max_exp_param = 0
	purity = 0.0
	while exp_param<=50:
		try:
			multi_view_kmeans(X_mats, n_images, n_class, exp_param)
			purity = test_cluster_accuracy()
			print purity
		except Exception as e:
			print exp_param
		
		if max_purity<purity:
			max_purity = purity
			max_exp_param = exp_param
		exp_param += 2
	print max_purity
	print max_exp_param

if __name__ == "__main__":
	print 'multi-view kmeans'
	
	n_class = 130
	n_images = 10000
	exp_param = 48
	file_name = "data/UI_1kPos_n0.txt"
	dataset_load = DatasetLoad(file_name)
	data_mat = dataset_load.load_data_mat()
	X_mats = load_multi_data(data_mat)

	multi_view_kmeans(X_mats, n_images, n_class, exp_param)
	#vilidation_test(X_mats, n_images, n_class, exp_param)
	#print test_cluster_accuracy()


