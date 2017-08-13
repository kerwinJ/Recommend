import numpy as np

def euclidean_dist(vector1, vector2):
	return 1/np.sum((vector1-vector2)**2)

def jaccard_dist(vec1_dict, vec2_dict):
		vector1Set = set(vec1_dict)
		vector2Set = set(vec2_dict)
		interSet = len(vector1Set & vector2Set)
		unionSet = len(vector1Set | vector2Set)
		if unionSet == 0:
			return 0
		else:
			return float(interSet)/(unionSet)

if __name__ == "__main__":
	print 'hello'