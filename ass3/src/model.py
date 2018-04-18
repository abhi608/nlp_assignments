from __future__ import division
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable as Var

dtype = torch.FloatTensor
if torch.cuda.device_count() > 0:
    dtype = torch.cuda.FloatTensor

class Model(nn.Module):
	def __init__(self, word_vocab, pos_vocab, rel_vocab, embedding_size=50, classes=3):
		super(Model, self).__init__()
		self.word_vocab = word_vocab
		self.pos_vocab = pos_vocab
		self.rel_vocab = rel_vocab
		self.embedding_size = embedding_size
		self.classes = classes
		self.d = 18

		self.fc1 = nn.Linear(self.word_vocab * self.d, self.embedding_size, bias=False)  # w1_w
		self.fc2 = nn.Linear(self.pos_vocab * self.d, self.embedding_size, bias=False)  # w1_t
		self.fc3 = nn.Linear(self.rel_vocab * self.d, self.embedding_size, bias=False)  # w1_l4
		self.fc4 = nn.Linear(self.embedding_size, self.classes, bias=False)  # w1_l4
		self.softmax = nn.Softmax(dim=0)

	def forward(self, word, pos, rel):
		# print "word shape: ", word.shape
		e1 = self.fc1(word)
		e2 = self.fc2(pos)
		e3 = self.fc3(rel)
		e = e1 + e2 + e3
		activ = torch.pow(e, 3) #1 x embedding_size
		y = self.fc4(activ) #1 x 3
		final = self.softmax(y.view(self.classes))
		return final.view(1, self.classes)


