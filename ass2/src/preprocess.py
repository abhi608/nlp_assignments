import os

indir = ["./aclImdb/train/pos/", "./aclImdb/train/neg/", "./aclImdb/test/pos/", "./aclImdb/test/neg/"]
outdir = ["./data/train/pos/", "./data/train/neg/", "./data/test/pos/", "./data/test/neg/"]
delete_list = ["<br />"]

for i in range(len(indir)):
	from_dir = indir[i]
	to_dir = outdir[i]
	for filename in os.listdir(from_dir):
		if filename.endswith(".txt"):
			fin = open(from_dir+filename)
			fout = open(to_dir+filename, "w+")
			for line in fin:
			    for word in delete_list:
			        line = line.replace(word, "")
			    fout.write(line)
			fin.close()
			fout.close()


