"""
This file contains code for evaluating the trained event2vec event-emeddding. It does so
by calculating patient similarity between a pair of patients whose representations are obtained
by stacking all event column vectors in their respective patient paragraph into patient matrixes.
Here we assume that two patients are considered to be similar if they share at least one common
diagnosed disease. The final result is a scalar showing the percentage of correctly paired patients.
"""

import numpy as np
import json
import argparse
from scipy.spatial.distance import pdist, squareform

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_dim", default=50, type=int, help="Size of word embedding.")
parser.add_argument("--eval_method", default='DCOV', type=str, choices=['RV', 'DCOV'], help="options ['RV', 'DCOV'] ")
args = parser.parse_args()
print(args)


def get_patient_matrix(events_list):
	matrix = embedded[int(events_list[0])]
	for e in events_list[1:]:
		matrix = np.column_stack((matrix,embedded[int(e)]))
	return matrix

def calculate_similarity(p1, p2):

	if args.eval_method == 'RV':
		# RV coefficient
		m1 = all_patients[p1]
		m2 = all_patients[p2]
		XX_t = m1.dot(m1.T)
		YY_t = m2.dot(m2.T)
		return np.trace(XX_t.dot(YY_t))/np.sqrt(XX_t.trace()**2 * YY_t.trace()**2)


	elif args.eval_method=='DCOV':
		# DCOV 
		X = all_patients[p1]
		Y = all_patients[p2]
		n = X.shape[0]
		if Y.shape[0] != X.shape[0]:
			raise ValueError('Number of samples must match')
		a = squareform(pdist(X))
		b = squareform(pdist(Y))
		A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
		B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

		dcov2_xy = (A * B).sum()/float(n * n)
		dcov2_xx = (A * A).sum()/float(n * n)
		dcov2_yy = (B * B).sum()/float(n * n)
		return np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))



# Load lookup table
embedded = np.loadtxt('embedded_%d.txt'%args.embedding_dim)

with open('patients_paragraph_event.txt') as eventfile:
	patients = eventfile.readlines()

with open('patient_diagnosis.json') as f:
	patients_diagnosis = json.load(f)


all_patients = {}

for p in patients:
	patient = p.split(' ')
	all_patients[patient[0]] = get_patient_matrix(patient[1:])

count = 0
total = 0
threshold = 0.8

true_pos = 0.
true_neg = 0.
false_pos = 0.
false_neg = 0.

for p1 in all_patients:
	for p2 in all_patients:
		truth =  True if len(set(patients_diagnosis[p1]).intersection(set(patients_diagnosis[p2]))) > 0 else False

		sim = calculate_similarity(p1, p2)

		if truth and sim> threshold:
			true_pos +=1
		elif truth and not(sim>threshold):
			false_neg +=1
		elif not truth and sim>threshold:
			false_pos +=1
		elif not truth and not(sim>threshold):
			true_neg +=1

		if (sim > threshold) == truth:
			count+=1
		total+= 1



p = 0 if (true_pos + false_pos)==0 else true_pos/(true_pos + false_pos)
r = true_pos/(true_pos + false_neg)
print("precision: ", p)
print("recall: ", r)
print("F1: ", 0 if (r+p)==0 else 2* (r*p)/(r+p))
print("Evaluation score: ", float(count)/total)