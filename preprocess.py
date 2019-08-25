""""
This file preprocesses patient paragraph for skip-gram training input by producing list of
(target_event, [context_event timestamp]). This is done by sliding a window of varying size for
each medical event based on frequency of occurrence of that event in that given patient. The main
idea is that patients with chronic disease have a relatively frequent medical events that depends
on other context events over a relatively long window size. The reverse is true for acute diseases.

length of window for event i and patient p is given by (a and θ are constants) :
	
		L(i, p) = f(i, p) ∗ a + b
"""
import numpy as np
from nltk import word_tokenize, FreqDist

a = 2
b = 0

with open('patients_paragraph_event.txt') as eventfile, open('patients_paragraph_time.txt') as timefile:
	events = eventfile.readlines()
	times = timefile.readlines()

trX = []
trY = []
trT = []

for p in range(len(events)):
	event_seq = events[p].split(' ')
	time_seq = times[p].split(' ')
	fdist = FreqDist( word_tokenize(events[p]))

	for pos, target in enumerate(event_seq[1:]):
		#window = fdist[target]*a + b
		window=5

		for i in range(window):
			if pos+i+1 >= len(event_seq):
				break
			trX.append(int(target))
			trY.append(int(event_seq[pos+i+1]))
			trT.append(float(time_seq[pos+i+1]) - float(time_seq[pos+1]))

		for i in range(window):
			if pos-i+1 < 1:
				break
			trX.append(int(target))
			trY.append(int(event_seq[pos-i+1]))
			trT.append(float(time_seq[pos-i+1]) - float(time_seq[pos+1]))



print('trX', len(trX))
print('trY', len(trY))
print('trT', len(trT))
np.save('target.npy', trX)
np.save('context.npy', trY)
np.save('timestamp.npy', trT)