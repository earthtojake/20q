import numpy as np 
import csv
import re

class Zoo(object):

	def __init__(self,src):

		self.questions = []
		self.targets = []
		self.qas = []

		self.src = src

		with open(self.src,'rb') as zoocsv:
			entries = list(csv.reader(zoocsv))
			labels = entries[0]
			self.questions = labels[1:]
			duplicates = 0
			for row in entries[1:]:
				tmp_qas = map(int,row[1:])
				if tmp_qas not in self.qas: # remove duplicate entries
					self.targets.append(row[0])
					self.qas.append(map(int,row[1:]))
				else:
					duplicates+=1
			print len(self.targets),'targets','(',duplicates,'duplicates removed )'
			print len(self.questions),'questions'

	def get_answer(self,_question,_target):
		for target in self.targets:
			if target == _target:
				for question in self.questions:
					if question == _question:
						return self.qas[self.targets.index(target)][self.questions.index(question)]
				break

		return None

	def get_answer_i(self,qi,ti):
		return self.qas[ti][qi]

	def animal_in_list(self,_target):
		if _target in self.targets:
			return True
		return False

	def test(self):
		target = raw_input('Enter animal: ')
		while (self.animal_in_list(target) == False):
			print 'Animal not in list. Please pick another animal...'
			target = raw_input('Enter animal: ')

		while True:
			question = raw_input('Question: ')
			answer = self.get_answer(question,target)
			if answer == 1:
				print 'Yes'
			elif answer == 0:
				print 'No'
			else:
				print 'Unknown question'

if __name__ == "__main__":
	z = Zoo('zoo.csv')
	z.test()
