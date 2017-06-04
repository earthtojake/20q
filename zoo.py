import numpy as np 
import csv
import re

# Zoo class is a bot to simulate a human user against the 20Q bot.
# It supplies the 20Q bot with questions and responds to these questions
# from the dataset to ensure accuracy. 

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
				if tmp_qas not in self.qas: # remove indistinguishable entries
					self.targets.append(row[0])
					self.qas.append(map(int,row[1:]))
				else:
					duplicates+=1

			print '\nloading',src,'...'
			print len(self.targets),'targets','(',duplicates,'duplicates removed )'
			print len(self.questions),'questions'

	# get the answer from two strings
	def get_answer(self,_question,_target):
		for target in self.targets:
			if target == _target:
				for question in self.questions:
					if question == _question:
						return self.qas[self.targets.index(target)][self.questions.index(question)]
				break

		return None

	# get the answer from a pair of question and target indexes
	def get_answer_i(self,qi,ti):
		ans = self.qas[ti][qi]
		if ans == 1: return 1
		else: return -1

	def animal_in_list(self,_target):
		if _target in self.targets:
			return True
		return False

	# test the zoo bot to verify its reliability
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
	z = Zoo('big.csv')
	z.test()
