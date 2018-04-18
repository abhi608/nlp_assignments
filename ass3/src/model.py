from __future__ import division
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable as Var
from torch.nn.parameter import Parameter

# dtype = torch.FloatTensor
# if torch.cuda.device_count() > 0:
#     dtype = torch.cuda.FloatTensor

class Model(nn.Module):
	def __init__(self, word_vocab, pos_vocab, rel_vocab, embedding_size=50, classes=3, vocab=None):
		super(Model, self).__init__()
		print "#classes: ", classes
		self.word_vocab = word_vocab
		self.pos_vocab = pos_vocab
		self.rel_vocab = rel_vocab
		self.embedding_size = embedding_size
		self.classes = classes
		self.d = 18
		self.vocab = vocab

		self.fc1 = nn.Linear(self.word_vocab * self.d, self.embedding_size, bias=False)  # w1_w
		self.fc2 = nn.Linear(self.pos_vocab * self.d, self.embedding_size, bias=False)  # w1_t
		self.fc3 = nn.Linear(self.rel_vocab * self.d, self.embedding_size, bias=False)  # w1_l
		self.fc4 = nn.Linear(self.embedding_size, self.classes, bias=False)  # w2

		print "word2vec started"
		tmp_weight = []
		if self.vocab:
			with open("./glove/glove.6B.200d.txt", "rb") as lines:
				for j in range(self.d):
					tmp_weight.append(np.zeros((200)))
				w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
				for word in self.vocab:
					for j in range(self.d):
						if word in w2v:
							tmp_weight.append(w2v[word])
						else:
							tmp_weight.append(np.zeros((200)))

                tmp_weight = np.transpose(np.array(tmp_weight))
                print "tmp_weight: ", tmp_weight.shape, self.word_vocab * self.d, self.embedding_size
                self.fc1.weight.data.copy_(torch.from_numpy(tmp_weight))


		tmp = np.random.uniform(low=-0.01, high=0.01, size=(self.embedding_size, self.pos_vocab * self.d))
		self.fc2.weight.data.copy_(torch.from_numpy(tmp))

		tmp = np.random.uniform(low=-0.01, high=0.01, size=(self.embedding_size, self.rel_vocab * self.d))
		self.fc3.weight.data.copy_(torch.from_numpy(tmp))
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


