# coding: utf-8

from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os
import pickle
import json
from collections import OrderedDict

dataDir		='../../Data'
versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType ='train2014'
annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
imgDir 		= '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)
num_words   = 1000
# initialize VQA api for QA annotations
vqa=VQA(annFile, quesFile)

# load and display QA annotations for given question types
"""
All possible quesTypes for abstract and mscoco has been provided in respective text files in ../QuestionTypes/ folder.
"""

## load question types

Quest_Dir = "../QuestionTypes/mscoco_question_types.txt"
f=open(Quest_Dir, "r")
contents = f.read().split("\n")

if contents[-1] == "":
	contents = contents[:-1]

Train_Dir = "../../Data/Images/train.pickle"

coco_train = pickle.load(open(Train_Dir, "rb"))

ans_count = {}

i = 0
for cont in contents:
	print(cont)
	annIds = vqa.getQuesIds(quesTypes=cont)
	anns = vqa.loadQA(annIds)
	for ann in anns:
		#print(ann)
		anss = ann['answers']
		#print(anss)
		for ans in anss:
			i += 1
			an = ans['answer']
			#print(an)
			if an not in ans_count:
				ans_count[an] = 1
			else:
				ans_count[an] +=1
#print(ans_count)
ans_count2 = [(k,v) for k, v in sorted(ans_count.items(), key=lambda item: item[1], reverse = True)]
ans_count3 = ans_count2[0:num_words]
ans_count3 = {k:v for (k,v) in ans_count3}


#print(i)
#print(ans_count3)
jsave = json.dumps(ans_count3)
f = open("1000_words.json","w")
f.write(jsave)
f.close()

with open('1000_words.json') as g:
		data = json.load(g)
#print(data)

gt_mat = np.zeros((1,1000))
