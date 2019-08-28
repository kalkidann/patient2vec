"""
This file contains code for training a skip-gram based event2vec embedding that also considers the timestamps
between successive medical events. Specifically in a single training sample of (target-event, context-event), 
the time interval from which the context-event deviates from the target-event is also appended to the context-event
to produce a training sample of (target-event, [context-event, timestamp]). The main intuition is that inorder to
learn a good embedding vector for the target-event, importance is attached not only to the surrounding context-event
but also to the time difference between the two events.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print('[' + display_now + ']' + ' ' + msg)

# load data
train_size = None
def load_data():
	global train_size
	trX = np.load('target.npy')
	trY = np.load('context.npy')
	trT = np.load('timestamp.npy')
	train_size = len(trX)
	return trX, trY, trT


def get_generator(batch_size):
	trX, trY, trT = load_data()
	trX = torch.tensor(trX, dtype=torch.long)
	trY = torch.tensor(trY, dtype=torch.long)
	trT = torch.tensor(trT, dtype=torch.float)

	random_indexs = np.random.randint(len(trY), size=len(trY))

	current = 0
	while current < len(trY):
		end = current + batch_size
		if end > len(trY):
			end=len(trY)

		yield (trX[current:end], trY[current:end], trT[current:end])
		current +=batch_size


# Model
class Model(nn.Module):
	def __init__(self, inout_dim, embedding_dim):
		super(Model, self).__init__()
		self.inout_dim = inout_dim
		self.embedding_dim = embedding_dim


		self.embedding = nn.Embedding(inout_dim, embedding_dim)
		self.fc = nn.Linear(embedding_dim, inout_dim+1)

	def forward(self, event):

		embedded = self.embedding(event)
		probs = torch.softmax(self.fc(embedded)[:,0:-1], 1) 
		time = embedded[:,-1].view(-1,1)
		return torch.cat([probs, time], 1)




def train(epoch, model, iterator, optimizer, criterion):
    model.train()

    for i, (inputt, label, time) in enumerate(iterator):

        inputt = inputt.to(device)
        label = label.to(device)
        time = time.to(device)

        optimizer.zero_grad()
        predictions = model(inputt)
        loss = criterion(predictions, torch.cat( [torch.tensor(F.one_hot(label, inout_dim), dtype=torch.float, device=device) , time.view(-1, 1)], 1) ) 
        loss.backward(torch.ones_like(loss))
        optimizer.step()

        if i % DISPLAY_FREQ == 0:
            msg = "Epoch %02d, Iter [%03d/%03d]" % (
                epoch, i, train_size/BATCH_SIZE
            )
            LOG_INFO(msg)



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--dictionary_size", default=753, type=int, help="Dictionary size")
	parser.add_argument("--epochs", default=100, type=int, help="Number of epoch.")
	parser.add_argument("--embedding_dim", default=50, type=int, help="Size of word embedding.")
	parser.add_argument("--batch_size", default=64, type=int, help="Batch size to use during training.")
	parser.add_argument("--display_freq", default=100, type=int, help="Display frequency")
	parser.add_argument("--lr", default=0.1, type=float, help="Learning rate for optimizer")
	parser.add_argument("--copy_model", default=False, type=bool, choices=[True, False], help="options")
	parser.add_argument("--save_model", default=False, type=bool, choices=[True, False], help="options")

	args = parser.parse_args()
	print(args)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	inout_dim = args.dictionary_size
	embedding_dim = args.embedding_dim

	model = Model(inout_dim, embedding_dim)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	criterion = nn.MSELoss(size_average=None, reduce=None, reduction='none')


	model = model.to(device)
	criterion = criterion.to(device)


	if args.copy_model:
		model.load_state_dict(torch.load('model_%d.pth'%embedding_dim))

	DISPLAY_FREQ = args.display_freq
	MAX_EPOCH = args.epochs
	BATCH_SIZE = args.batch_size
	for epoch in range(1, MAX_EPOCH + 1):
		train_iterator = get_generator(BATCH_SIZE)
		train(epoch, model, train_iterator, optimizer, criterion)

	if args.save_model:
		torch.save(model.state_dict(), 'model_%d.pth'%embedding_dim)
		np.savetxt('embedded_%d.txt'% embedding_dim, model.embedding.weight.detach().cpu().numpy())
