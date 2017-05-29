import numpy as np 
import csv
import re

def get_questions(src):
	with open(src,'rb') as zoocsv:
		labels = list(csv.reader(zoocsv))[0]
		return labels[1:]

def get_targets(src):
	targets = []
	with open(src,'rb') as zoocsv:
		for row in list(csv.reader(zoocsv))[1:]:
			targets.append(row[0])
	return targets

def get_answer(src,question,target):

	qs = get_questions(src)

	with open(src,'rb') as zoocsv:
		reader = csv.reader(zoocsv)
		for row in reader:
			if row[0] == target:
				for q in qs:
					if q == question:
						answer = row[qs.index(q)+1]
						return int(answer)
				break

	return None

def animal_in_list(src,target):
	with open(src,'rb') as zoocsv:
		reader = csv.reader(zoocsv)
		for row in reader:
			if row[0] == target:
				return True
	return False


if __name__ == "__main__":
	target = raw_input('Enter animal: ')
	while (animal_in_list(target) == False):
		print 'Animal not in list. Please pick another animal...'
		target = raw_input('Enter animal: ')

	while True:
		question = raw_input('Question: ')
		answer = get_answer(question,target)
		if answer == 1:
			print 'Yes'
		elif answer == 0:
			print 'No'
		else:
			print 'Unknown question'

