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
			if cnt > 5:
				break
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

# print len(word_vocabulary), len(pos_vocabulary), len(rel_vocabulary)

for i in range(len(corpus)):
	sigma = []
	beta = []
	idx_to_word = {}
	head = {}
	child = {}
	# leftmost = {}
	# rightmost = {}
	for j in range(len(corpus[i])):
		idx_to_word[corpus[i][j][0]] = corpus[i][j][1]
		beta.append(corpus[i][j][0])
		head[corpus[i][j][0]] = corpus[i][j][6]
		if corpus[i][j][6] not in child:
			child[corpus[i][j][6]] = [int(corpus[i][j][0])]
		else:
			child[corpus[i][j][6]].append(int(corpus[i][j][0]))
		# if corpus[i][j][6] not in leftmost or int(leftmost[corpus[i][j][6]]) > int(corpus[i][j][0]):
		# 	leftmost[corpus[i][j][6]] = corpus[i][j][0]
		# if corpus[i][j][6] not in rightmost or int(rightmost[corpus[i][j][6]]) < int(corpus[i][j][0]):
		# 	rightmost[corpus[i][j][6]] = corpus[i][j][0]

	for key in child:
		child[key].sort()

	sigma.append('-1') #-1 = ROOT
	transitions = []
	count = 0
	x_word_main = []
	x_pos_main = []
	x_rel_main = []
	y_cur = []
	while len(beta) > 0:
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
				# print beta[0]
				# print "RIGHT"
			else:
				action = 0
				tmp = beta.pop(0)
				sigma.insert(0, tmp)
				# print "SHIFT-1"
		count += 1
		y_cur.append(action)
		# print x_word
		# print sigma
		# print beta
		# print head
		# print "X_word: ", x_word[17], len(x_word)
		# print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
		x_word_main.append(x_word)
	print len(y_cur), len(x_word_main)
	print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"







