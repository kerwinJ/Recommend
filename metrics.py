import numpy as np 

'''
def cal_precision(idx, rank, rows_ground):

	row_ground = (rows_ground[idx]-1) // 100
	accNo = 0.0
	for i in xrange(len(rank)):
		if rank[i][0]//100 == row_ground:
			accNo += 1
	return accNo/len(rank) 
	
def cal_recall(idx, rank, rat_no, rows_ground):

	row_ground = (rows_ground[idx]-1) // 100
	accNo = 0.0
	for i in xrange(len(rank)):
		if rank[i][0]//100 == row_ground:
			accNo += 1
	return accNo/(100-rat_no)

def cal_f1_score(precision, recall):
	f1_score = 2*precision*recall/(precision+recall)
	return f1_score
'''

def cal_precision(idx, rank, rows_ground):
	row_ground = (rows_ground[idx]-1) // 100
	topk = len(rank)
	precisions = np.zeros(topk)
	for k in range(topk, 0, -1):
		accNo = 0.0
		for i in xrange(k):
			if rank[i][0]//100 == row_ground:
				accNo += 1
		precisions[k-1] = accNo/k
	return precisions

def cal_recall(idx, rank, rat_no, rows_ground):

	row_ground = (rows_ground[idx]-1) // 100
	topk =len(rank)
	recalls = np.zeros(topk)
	for k in range(topk, 0, -1):
		accNo = 0.0
		for i in xrange(k):
			if rank[i][0]//100 == row_ground:
				accNo += 1
		recalls[k-1] = accNo/(100-rat_no)
	return recalls

def cal_f1_score(precisions, recalls):

	topk = len(precisions)
	f1_scores = np.zeros(topk)
	for k_temp in range(topk, 0, -1):
		k = k_temp-1
		f1_scores[k] = 2*precisions[k]*recalls[k]/(precisions[k]+recalls[k])

	return f1_scores

if __name__ == '__main__':
	print ''
	