import re

file = open("../test.txt", "r")
text = file.read()
file.close()

# Handle all the sentences of the form 'hello there' as a single entity-- starting with a lowercase alphabet
regex1 = r"((,|;)\s)(')([a-z](\w|\s|\?' |\,|\.|\(|\)|\-|\;|'\w|[a-z]\?\s[A-Z]|'(\w|\s)*,*'|[a-z] \? [A-Z])*?(\.|\-|\?|\!))(')"
# Handle all other active voice by independently handling opening and closing single quotes
regex2 = r"([\.|,|\!|\?|\-]\s?)(')"
# Handle the form < end. 'Hello people' >
regex3 = r"([^a-z]{,2})(')([A-Z])"
# Remove false positives
regex4 = r"(\s'[\w|\s]+,)(\")"
# Remove false positives
regex5 = r"(\")([\w]+';)"

subst1 = r'\1"\4"'
subst2 = r'\1"'
subst3 = r'\1"\3'
subst4 = r"\1'"
subst5 = r"'\2"
new_text = re.sub(regex1, subst1, text, 0, re.MULTILINE)
if new_text:
	# print new_text
	new_text = re.sub(regex2, subst2, new_text, 0, re.MULTILINE)
	if new_text:
		# print new_text
		new_text = re.sub(regex3, subst3, new_text, 0, re.MULTILINE)
		if new_text:
			# print new_text
        		new_text = re.sub(regex4, subst4, new_text, 0, re.MULTILINE)
                        if new_text:
                            # print(new_text)
                            new_text = re.sub(regex5, subst5, new_text, 0, re.MULTILINE)
                            if new_text:
                                print(new_text)
                                file = open("new_test.txt", "w")
                                file.write(new_text)
                                file.close()
                            else:
                                print("Error in regex5!")
                        else:
                            print("Error in regex4!")
		else:
			print("Error in regex3!")
	else:
		print("Error in regex2!")
else:
	print("Error in regex1!")

