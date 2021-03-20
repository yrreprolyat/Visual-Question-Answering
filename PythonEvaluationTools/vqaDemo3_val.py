# coding: utf-8

from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os
import pickle
import json
from collections import OrderedDict
import numpy as np
from tqdm import tqdm


dataDir		='../../Data'
versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType ='val2014'
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

Eval_Dir = "../../Data/Images/val.pickle"

coco_train = pickle.load(open(Eval_Dir, "rb"))


with open('1000_words.json') as g:
		ans_count3 = json.load(g)
#print(ans_count3)


ans_gt_dict = {k:i for i, k in enumerate(ans_count3)}

jsave0 = json.dumps(ans_gt_dict)
h = open("ans_gt_dict_val.json","w")
h.write(jsave0)
h.close()

#gt_mat = np.zeros((4437570,num_words), dtype = int)

img_id_dict = {}
ques_id_dict = {}
ques_img_dict = {}
ques_vec_dict = {}

First = True
i =0
for cont in tqdm(contents):
	print(cont)
	annIds = vqa.getQuesIds(quesTypes=cont)
	anns = vqa.loadQA(annIds)

	for ann in anns:
		q_id = ann['question_id']
		img_id = ann['image_id']
		img_id_dict[i] = img_id
		ques_id_dict[i] = q_id
		ques_img_dict[q_id] = img_id
		anss = ann['answers']
		gt_dict = {}
		#gt_vec = np.zeros(num_words, dtype = int)
		for ans in anss:
			an = ans['answer']
			if an in ans_gt_dict:
				vec_idx = ans_gt_dict[an]
				if vec_idx in gt_dict:
					gt_dict[vec_idx] += 1
				else:
					gt_dict[vec_idx] = 1
				#gt_dict[vec_idx]
				#gt_vec[vec_idx] += 1 ##get the oriignal array
		ques_vec_dict[i] = gt_dict
		i +=1

print(i)
jsave = json.dumps(img_id_dict)
f = open("img_id_dict_val.json","w")
f.write(jsave)
f.close()

jsave2 = json.dumps(ques_id_dict)
f2 = open("ques_id_dict_val.json","w")
f2.write(jsave2)
f2.close()

jsave3 = json.dumps(ques_vec_dict)
f3 = open("ques_vec_dict_val.json","w")
f3.write(jsave3)
f3.close()

exit(0)
#gt_mat = gt_mat.astype(int)

print(gt_mat.shape)
np.savez("gt_mat.npy", gt_mat)
exit(0)




exit(0)





exit(0)
	#randomAnn = random.choice(anns)
	##print("HELLO1")
	#print(randomAnn)
quest = vqa.showQA([randomAnn])
print("This is quest")
print(quest)
imgId = randomAnn['image_id']
image_feat = coco_train[imgId]
print(image_feat)
print(image_feat.shape)
print(randomAnn['question_id'])
imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
if os.path.isfile(imgDir + imgFilename):
	I = io.imread(imgDir + imgFilename)
	plt.imshow(I)
	plt.axis('off')
	plt.show()

exit(0)
# load and display QA annotations for given answer types
"""
ansTypes can be one of the following
yes/no
number
other
"""
annIds = vqa.getQuesIds(ansTypes='yes/no');
anns = vqa.loadQA(annIds)
randomAnn = random.choice(anns)
print(randomAnn)
vqa.showQA([randomAnn])
print("HELLO2")
imgId = randomAnn['image_id']
print(imgId)
exit(0)
imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
if os.path.isfile(imgDir + imgFilename):

	I = io.imread(imgDir + imgFilename)
	plt.imshow(I)
	plt.axis('off')
	plt.show()

# load and display QA annotations for given images
"""
Usage: vqa.getImgIds(quesIds=[], quesTypes=[], ansTypes=[])
Above method can be used to retrieve imageIds for given question Ids or given question types or given answer types.
"""
ids = vqa.getImgIds()
annIds = vqa.getQuesIds(imgIds=random.sample(ids,5));
anns = vqa.loadQA(annIds)
randomAnn = random.choice(anns)
print("HELLO3")
print(randomAnn)
vqa.showQA([randomAnn])
imgId = randomAnn['image_id']
imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
if os.path.isfile(imgDir + imgFilename):

	I = io.imread(imgDir + imgFilename)
	plt.imshow(I)
	plt.axis('off')
	plt.show()
