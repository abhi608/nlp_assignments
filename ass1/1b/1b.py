import re

file = open("../test.txt", "r")
text = file.read()
file.close()

# Handle all the sentences that end with [.?!]
regex1 = r"(?<![A-Z]\.)(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)(\s)(\s*[A-Z\'])"
# Handle all sentences ending with .' | ?' | !'
regex2 = r"(?<![A-Z]\.)(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.'|\?'|\!')(\s)(\s*[A-Z\'])"
# For handling sentences present at the beginning of the file
regex3 = r"(?<!--)\n\n"
# For handling all the "--" in the text file 
regex4 = r"(?<=--)\n\n"

subst1 = "</s><s>"
subst = r'</s><s>\2'
subst2 = ""
new_text = re.sub(regex1, subst, text, 0, re.MULTILINE)
if new_text:
    new_text = re.sub(regex2, subst, new_text, 0, re.MULTILINE)
    if new_text:
        # print(new_text)
	new_text = re.sub(regex3, subst1, new_text, 0, re.MULTILINE)
	if new_text:
            new_text = re.sub(regex4, subst2, new_text, 0, re.MULTILINE)
            if new_text:
                print(new_text)
                file = open("../2/training_data.txt", "w")
                file.write(new_text)
        	file.close()
            else:
                print("Error in regex4!")
	else:
	    print("Error in regex3!")
    else:
    	print("Error in regex2!")
else:
    print("Error in regex1!")

