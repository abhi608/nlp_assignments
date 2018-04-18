import pickle
import torch
from torch.autograd import Variable as Var
from model import Model
from torch.autograd import Variable as Var


dtype = torch.FloatTensor
if torch.cuda.device_count() > 0:
    dtype = torch.cuda.FloatTensor

#-------------------loading data corpus--------------------------------------------
treebank = "./train.conll"
print "Loading treebank"
corpus = []
sentence = []
cnt = 0
with open (treebank, 'r') as f:
	for line in f:
		line = line.strip()
		if not line:
			corpus.append(sentence)
			sentence = []
			cnt += 1
			# print sentence
			# if cnt > 5:
			# 	break
		else:
			sentence.append(line.split('\t'))

word_vocabulary = []
pos_vocabulary = []
rel_vocabulary = []
for i in range(len(corpus)):
	for j in range(len(corpus[i])):
		if corpus[i][j][1] not in word_vocabulary:
			word_vocabulary.append(corpus[i][j][1])
		if corpus[i][j][3] not in pos_vocabulary:
			pos_vocabulary.append(corpus[i][j][3])
		if corpus[i][j][7] not in rel_vocabulary:
			rel_vocabulary.append(corpus[i][j][7])

print len(word_vocabulary), len(pos_vocabulary), len(rel_vocabulary)


model = Model(len(word_vocabulary)+1, len(pos_vocabulary)+1, len(rel_vocabulary)+1, embedding_size=50, classes=3)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for i in range(len(corpus)):
	sigma = []
	beta = []
	idx_to_word = {}
	idx_to_pos = {}
	head = {}
	child = {}
	idx_to_res = {}
	for j in range(len(corpus[i])):
		idx_to_word[corpus[i][j][0]] = corpus[i][j][1]
		idx_to_pos[corpus[i][j][1]] = corpus[i][j][3]
		idx_to_res[corpus[i][j][1]] = corpus[i][j][7]
		beta.append(corpus[i][j][0])
		head[corpus[i][j][0]] = corpus[i][j][6]
		if corpus[i][j][6] not in child:
			child[corpus[i][j][6]] = [int(corpus[i][j][0])]
		else:
			child[corpus[i][j][6]].append(int(corpus[i][j][0]))


	for key in child:
		child[key].sort()

	sigma.append('-1') #-1 = ROOT
	transitions = []
	count = 0
	x_word_main = []
	x_pos_main = []
	x_rel_main = []
	y_cur = []
	loss_final = 0.0
	mini_batch = 0
	while len(beta) > 0:
		mini_batch += 1
		x_word = []
		x_pos = []
		x_rel = []
		if len(sigma) >= 3:
			tmp = sigma[0]
			if tmp == '-1':
				cur = [0] * (len(word_vocabulary)+1)
				cur[0] = 1
				x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[tmp])+1] = 1
				x_word.append(cur)

			if tmp in child:
				word = str(child[tmp][0])
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #leftmost
				x_word.append(cur)
				word = str(child[tmp][-1])
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #rightmost
				x_word.append(cur)
				if len(child[tmp]) >= 2:
					word = str(child[tmp][1])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd leftmost
					x_word.append(cur)
					word = str(child[tmp][-2])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd rightmost
					x_word.append(cur)
				else:
					word = str(child[tmp][0])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd leftmost
					x_word.append(cur)
					word = str(child[tmp][-1])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd rightmost
					x_word.append(cur)
				leftest_word = str(child[tmp][0])
				if leftest_word in child:
					word = str(child[leftest_word][0])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #leftmost of leftmost
					x_word.append(cur)
				else:
					cur = [0] * (len(word_vocabulary)+1)
					x_word.append(cur)
				rightest_word = str(child[tmp][-1])
				if rightest_word in child:
					word = str(child[rightest_word][-1])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #rightmost of rightmost
					x_word.append(cur)
				else:
					cur = [0] * (len(word_vocabulary)+1)
					x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)

			tmp = sigma[1]
			if tmp == '-1':
				cur = [0] * (len(word_vocabulary)+1)
				cur[0] = 1
				x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[tmp])+1] = 1 
				x_word.append(cur)

			if tmp in child:
				word = str(child[tmp][0])
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #leftmost
				x_word.append(cur)
				word = str(child[tmp][-1])
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #rightmost
				x_word.append(cur)
				if len(child[tmp]) >= 2:
					word = str(child[tmp][1])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd leftmost
					x_word.append(cur)
					word = str(child[tmp][-2])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd rightmost
					x_word.append(cur)
				else:
					word = str(child[tmp][0])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd leftmost
					x_word.append(cur)
					word = str(child[tmp][-1])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd rightmost
					x_word.append(cur)
				leftest_word = str(child[tmp][0])
				if leftest_word in child:
					word = str(child[leftest_word][0])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #leftmost of leftmost
					x_word.append(cur)
				else:
					cur = [0] * (len(word_vocabulary)+1)
					x_word.append(cur)
				rightest_word = str(child[tmp][-1])
				if rightest_word in child:
					word = str(child[rightest_word][-1])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #rightmost of rightmost
					x_word.append(cur)
				else:
					cur = [0] * (len(word_vocabulary)+1)
					x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)


			tmp = sigma[2]
			if tmp == '-1':
				cur = [0] * (len(word_vocabulary)+1)
				cur[0] = 1
				x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[tmp])+1] = 1 
				x_word.append(cur)



		elif len(sigma) >= 2:
			tmp = sigma[0]
			if tmp == '-1':
				cur = [0] * (len(word_vocabulary)+1)
				cur[0] = 1
				x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[tmp])+1] = 1 
				x_word.append(cur)

			if tmp in child:
				word = str(child[tmp][0])
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #leftmost
				x_word.append(cur)
				word = str(child[tmp][-1])
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #rightmost
				x_word.append(cur)
				if len(child[tmp]) >= 2:
					word = str(child[tmp][1])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd leftmost
					x_word.append(cur)
					word = str(child[tmp][-2])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd rightmost
					x_word.append(cur)
				else:
					word = str(child[tmp][0])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd leftmost
					x_word.append(cur)
					word = str(child[tmp][-1])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd rightmost
					x_word.append(cur)
				leftest_word = str(child[tmp][0])
				if leftest_word in child:
					word = str(child[leftest_word][0])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #leftmost of leftmost
					x_word.append(cur)
				else:
					cur = [0] * (len(word_vocabulary)+1)
					x_word.append(cur)
				rightest_word = str(child[tmp][-1])
				if rightest_word in child:
					word = str(child[rightest_word][-1])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #rightmost of rightmost
					x_word.append(cur)
				else:
					cur = [0] * (len(word_vocabulary)+1)
					x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)

			tmp = sigma[1]
			if tmp == '-1':
				cur = [0] * (len(word_vocabulary)+1)
				cur[0] = 1
				x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[tmp])+1] = 1 
				x_word.append(cur)

			if tmp in child:
				word = str(child[tmp][0])
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #leftmost
				x_word.append(cur)
				word = str(child[tmp][-1])
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #rightmost
				x_word.append(cur)
				if len(child[tmp]) >= 2:
					word = str(child[tmp][1])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd leftmost
					x_word.append(cur)
					word = str(child[tmp][-2])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd rightmost
					x_word.append(cur)
				else:
					word = str(child[tmp][0])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd leftmost
					x_word.append(cur)
					word = str(child[tmp][-1])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd rightmost
					x_word.append(cur)
				leftest_word = str(child[tmp][0])
				if leftest_word in child:
					word = str(child[leftest_word][0])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #leftmost of leftmost
					x_word.append(cur)
				else:
					cur = [0] * (len(word_vocabulary)+1)
					x_word.append(cur)
				rightest_word = str(child[tmp][-1])
				if rightest_word in child:
					word = str(child[rightest_word][-1])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #rightmost of rightmost
					x_word.append(cur)
				else:
					cur = [0] * (len(word_vocabulary)+1)
					x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)

			cur = [0] * (len(word_vocabulary)+1)
			x_word.append(cur)

		elif len(sigma) >= 1:
			tmp = sigma[0]
			if tmp == '-1':
				cur = [0] * (len(word_vocabulary)+1)
				cur[0] = 1
				x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[tmp])+1] = 1 
				x_word.append(cur)

			# print "child: ", child

			if tmp in child:
				word = str(child[tmp][0])
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #leftmost
				x_word.append(cur)
				word = str(child[tmp][-1])
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #rightmost
				x_word.append(cur)
				if len(child[tmp]) >= 2:
					word = str(child[tmp][1])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd leftmost
					x_word.append(cur)
					word = str(child[tmp][-2])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd rightmost
					x_word.append(cur)
				else:
					word = str(child[tmp][0])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd leftmost
					x_word.append(cur)
					word = str(child[tmp][-1])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #2nd rightmost
					x_word.append(cur)
				leftest_word = str(child[tmp][0])
				if leftest_word in child:
					word = str(child[leftest_word][0])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #leftmost of leftmost
					x_word.append(cur)
				else:
					cur = [0] * (len(word_vocabulary)+1)
					x_word.append(cur)
				rightest_word = str(child[tmp][-1])
				if rightest_word in child:
					word = str(child[rightest_word][-1])
					cur = [0] * (len(word_vocabulary)+1)
					cur[word_vocabulary.index(idx_to_word[word])+1] = 1 #rightmost of rightmost
					x_word.append(cur)
				else:
					cur = [0] * (len(word_vocabulary)+1)
					x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)
				x_word.append(cur)



			cur = [0] * (len(word_vocabulary)+1)
			x_word.append(cur)
			x_word.append(cur)
			cur = [0] * (len(word_vocabulary)+1)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)

		elif len(sigma) >= 0:
			cur = [0] * (len(word_vocabulary)+1)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)


		if len(beta) >= 3:
			tmp = beta[0]
			if tmp == '-1':
				cur = [0] * (len(word_vocabulary)+1)
				cur[0] = 1
				x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				# print word_vocabulary.index(idx_to_word[tmp])
				cur[word_vocabulary.index(idx_to_word[tmp])+1] = 1 
				x_word.append(cur)

			tmp = beta[1]
			if tmp == '-1':
				cur = [0] * (len(word_vocabulary)+1)
				cur[0] = 1
				x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[tmp])+1] = 1 
				x_word.append(cur)

			tmp = beta[2]
			if tmp == '-1':
				cur = [0] * (len(word_vocabulary)+1)
				cur[0] = 1
				x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[tmp])+1] = 1 
				x_word.append(cur)

		elif len(beta) >= 2:
			tmp = beta[0]
			if tmp == '-1':
				cur = [0] * (len(word_vocabulary)+1)
				cur[0] = 1
				x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[tmp])+1] = 1 
				x_word.append(cur)

			tmp = beta[1]
			if tmp == '-1':
				cur = [0] * (len(word_vocabulary)+1)
				cur[0] = 1
				x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[tmp])+1] = 1 
				x_word.append(cur)

			cur = [0] * (len(word_vocabulary)+1)
			x_word.append(cur)

		elif len(beta) >= 1:
			tmp = beta[0]
			if tmp == '-1':
				cur = [0] * (len(word_vocabulary)+1)
				cur[0] = 1
				x_word.append(cur)
			else:
				cur = [0] * (len(word_vocabulary)+1)
				cur[word_vocabulary.index(idx_to_word[tmp])+1] = 1 
				x_word.append(cur)

			cur = [0] * (len(word_vocabulary)+1)
			x_word.append(cur)
			x_word.append(cur)

		elif len(beta) >= 0:
			cur = [0] * (len(word_vocabulary)+1)
			x_word.append(cur)
			x_word.append(cur)
			x_word.append(cur)
		

		if count == 0:
			action = 0
			tmp = beta.pop(0)
			sigma.insert(0, tmp)
			# print "SHIFT"

		elif sigma[0] in head and head[sigma[0]] == beta[0]:
			action = 1
			# print sigma[0]
			head.pop(sigma[0], None)
			sigma.pop(0)	
			# print "LEFT"

		else:
			tmp1 = False
			for key in head:
				if head[key] == beta[0]:
					tmp1 = True
					break
			if tmp1 == False and head[beta[0]] == sigma[0]:
				action = 2
				head.pop(beta[0], None)
				beta.pop(0)
				tmp = sigma.pop(0)
				beta.insert(0, tmp)
				# print "RIGHT"
			else:
				action = 0
				tmp = beta.pop(0)
				sigma.insert(0, tmp)
				# print "SHIFT-1"
		count += 1
		y_cur.append(action)

		for k in range(len(x_word)):
			try:
				idx = x_word[k].index(1)
				if idx != 0:
					tmp_word = word_vocabulary[idx-1]
					tmp_pos = idx_to_pos[tmp_word]
					pos = [0] * (len(pos_vocabulary)+1)
					pos[pos_vocabulary.index(tmp_pos)+1] = 1
					x_pos.append(pos)
				else:
					pos = [0] * (len(pos_vocabulary)+1)
					pos[idx] = 1
					x_pos.append(pos)
			except:
				pos = [0] * (len(pos_vocabulary)+1)
				x_pos.append(pos)

		for k in range(len(x_word)):
			try:
				idx = x_word[k].index(1)
				if idx != 0:
					tmp_word = word_vocabulary[idx-1]
					tmp_res = idx_to_res[tmp_word]
					res = [0] * (len(rel_vocabulary)+1)
					res[rel_vocabulary.index(tmp_res)+1] = 1
					x_rel.append(res)
				else:
					res = [0] * (len(rel_vocabulary)+1)
					res[idx] = 1
					x_rel.append(res)
			except:
				res = [0] * (len(rel_vocabulary)+1)
				x_rel.append(res)

		#------------------------------Train neural net-------------------------------------------
		x_word_tensor = Var(torch.FloatTensor(x_word)).view(1,-1)
		x_pos_tensor = Var(torch.FloatTensor(x_pos)).view(1,-1)
		x_rel_tensor = Var(torch.FloatTensor(x_rel)).view(1,-1)
		y_desired = [action]
		y_desired = Var(torch.LongTensor(y_desired))
		optimizer.zero_grad()
		y_predicted = model(x_word_tensor, x_pos_tensor, x_rel_tensor)
		loss = criterion(y_predicted, y_desired)
		loss_final += loss.data[0]
		loss.backward()
       	optimizer.step()
	print "datapoint: ", str(i), "|", "loss: ", loss_final/mini_batch







