import zoo
import random
from neural import NeuralNetwork,NeuronLayer,Neuron
import math
from copy import copy
import progressbar
import sys
import numpy as np

# A class to store a 20Q round
# Takes in a target (correct answer) and nueral network
# Autoplay: generates an input vector for training / testing the network
# Play: interactive game between user and bot

class Q20(object):

	def __init__(self,target,nn):
		self.qs_asked = [] # indexes of qs_asked
		self.input_vector = [0]*len(qs)*2 # answers of all questions, answered or otherwise, initialised to 0 (input_vector)
		self.target = target # index of target animal set by user 
		self.nn = nn

	# select next question randomly and remove it from the list for this round
	def next_q(self,user=False):

		if len(self.qs_asked) == QUESTION_LIMIT or len(self.qs_asked) == len(qs): return -1

		q = random.choice(qs)
		while qs.index(q) in self.qs_asked: # wait for a question that hasn't been asked
			q = random.choice(qs) # select random question from list

		self.qs_asked.append(qs.index(q)) # add index of q-no to list of asked questions

		if user:
			print q

			yesno = raw_input("Answer (y/n): ")
			while yesno != 'y' and yesno != 'n':
				print 'Invalid input, y = yes, n = no'
				yesno = raw_input("Answer (y/n): ")

			if yesno == 'y': ans = 1
			elif yesno == 'n': ans = 0

		else: ans = zoo.get_answer(SRC,q,self.target) # returns 1 for yes, 0 for no
		qi = 2*qs.index(q) + ans # question index of q-no or q-yes answer

		self.input_vector[qi] = 1

		return qi

	# asks question, returns input vector
	def ask_q(self,qi):
		if qi not in self.qs_asked:
			self.qs_asked.append(qi)

		ans = zoo.get_answer(SRC,qs[qi],self.target)
		qii = 2*qi + ans

		self.input_vector[qii] = 1

	# guess target string
	def guess(self,target_guess,user=False):

		if user: print '\nBot guess =',target_guess
		if target_guess == self.target:
			if user: print 'Correct!\n'
			return 1
		else:
			if user: print 'Incorrect!\n'
			return -1

	def play(self,user=False):

		if user: print '\n================ SELECT ANIMAL ================'
		if user: print '~Your animal will not be shared with the bot until it makes its final guess'
		if user: print '==================== '+str(QUESTION_LIMIT)+' Q ===================='

		if self.target == None:
			self.target = raw_input("Your animal is: ")
			while self.target not in ts:
				print 'Animal not in list. Select another...'
				self.target = raw_input("Your animal is: ")

		# play game
		while self.next_q(user=user) != -1:
			continue

if len(sys.argv) >= 2: SRC = sys.argv[1]
else: SRC = 'zoo_medium1.csv' 

if len(sys.argv) >= 3: TRAINING_SIZE = int(sys.argv[2])
else: TRAINING_SIZE = 1000

if len(sys.argv) >= 4: LEARNING_RATE = float(sys.argv[3])
else: LEARNING_RATE = 0.1

if len(sys.argv) >= 5: QUESTION_LIMIT = int(sys.argv[4])
else: QUESTION_LIMIT = 10

qs = zoo.get_questions(SRC)
ts = zoo.get_targets(SRC)

input_size = len(qs)*2
output_size = len(ts)
hidden_size = int(math.sqrt(input_size+output_size))+5

targets = [[1 if i == j else 0 for i in range(output_size)]
               for j in range(output_size)]

nn = NeuralNetwork(input_size,hidden_size,output_size,learning_rate=LEARNING_RATE,hidden_layer_bias=1,output_layer_bias=1)

print '\n======================================='
print 'NETWORK SUMMARY'
print '=======================================\n' 

print 'src =',SRC
print 'input =',input_size
print 'output =',output_size
print 'hidden =',hidden_size
print 'question limit =',QUESTION_LIMIT
print 'training set =',TRAINING_SIZE,'*',output_size,'=',TRAINING_SIZE*output_size
print 'learning rate =',LEARNING_RATE

print '\n======================================='
print 'TRAINING'
print '=======================================\n' 

# create training sets, 500 games for each of the 100 animals (20,000)
num_training = TRAINING_SIZE*len(ts)
print 'Training with',num_training,'sets...'
training_sets = []

bar = progressbar.ProgressBar(maxval=num_training,widgets=[progressbar.Bar('=','[',']'),' ',progressbar.Percentage()])
bar.start()

i = 0
interval = int(num_training/100)

training_sets = []

for __ in range(TRAINING_SIZE):

	for t in range(len(ts)):

		r = Q20(t,nn)
		r.play()

		input_vector = r.input_vector # questions + answers
		target_vector = targets[ts.index(t)] # correct guess

		nn.train(input_vector,target_vector)
		training_sets.append([input_vector,target_vector])

		try:
			if i % interval == 0:bar.update(i+interval)
		except:
			pass
		i+=1

print nn.calculate_total_error(training_sets)

bar.finish()

print '\n======================================='
print 'TESTING SET'
print '=======================================\n'

wins = 0
num_games = 100
for i in range(num_games):

	t = random.choice(ts)
	r = Q20(t,nn)

	output = [0]*output_size # initialise output for first q selection
	input_vector = [0]*input_size

	qs_order = []

	for m in range(QUESTION_LIMIT):

		best_qi = -1
		best_std = -1

		# print '\tSelecting q',m,'against',output

		for j in range(len(qs)):

			if j in r.qs_asked:
				continue # question has already been asked, ignore

	    	# test vectors for q-yes and q-no

			test_vector_no = copy(input_vector)
			test_vector_no[2*j] = 1
			test_vector_yes = copy(input_vector)
			test_vector_yes[2*j+1] = 1

	    	# test outputs for q-yes
			test_output_yes = nn.feed_forward(test_vector_yes)

			# test outputs for q-no
			test_output_no = nn.feed_forward(test_vector_no)

	    	# average of test outputs (assuming 50/50 chance of a yes-no q)
			test_diff_yes = [x-y for x,y in zip(test_output_yes,output)]
			test_diff_no = [x-y for x,y in zip(test_output_no,output)]

			test_std_yes = np.std(test_diff_yes)
			test_std_no = np.std(test_diff_no)

			# select the smaller standard deviation to avoid 'risky' question selection
			if test_std_yes < test_std_no: test_std = test_std_yes
			else: test_std = test_std_no

	    	# select the biggest standard deviation, i.e. biggest separation of output probabilities,
	    	# for the worst case answer in each question
			if best_qi == -1 or test_std >= best_std:
				best_std = test_std
				best_qi = j

			# print '\t\t',j,qs[j],test_std,'(',qs[best_qi],best_std,')'

		if best_qi == -1: break

		r.ask_q(best_qi)
		qs_order.append(best_qi)

		input_vector = r.input_vector
		output = nn.feed_forward(input_vector)

		if max(output) > 0.95:
			break

	guess_i = output.index(max(output))

	qs_str = []
	for qi in qs_order:
		qs_str.append(qs[qi])

	if guess_i == ts.index(t):
		wins+=1
		print 'Game',i+1,'win:',ts[guess_i],'...',len(qs_str),'qs:',' '.join(qs_str)
	else:
		print '\nGame',i+1,'LOSS:',ts[guess_i],'(',t,')','...',len(qs_str),'qs:',' '.join(qs_str),'\n'

print '======================================='
print 'Correctly guessed',wins,'/',num_games,'games'
print '======================================='

# t = raw_input('\n\nYour turn. Select an animal: ')
# r = Round(target=t)
# r.play(user=True)

# input_vector_before = r.input_vector
# outputs = nn.feed_forward(input_vector_before)
# guess_i = outputs.index(max(outputs))
# if guess_i == ts.index(t):
# 	print '\n===================================================='
# 	print 'WIN: Bot guessed',ts[guess_i],'correctly!'
# 	print '===================================================='
# else:
# 	print '\n===================================================='
# 	print 'LOSS: Bot guessed',ts[guess_i],'instead of',t,''
# 	print '===================================================='
