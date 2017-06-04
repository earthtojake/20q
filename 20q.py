from zoo import Zoo
import random
from neural import NeuralNetwork,NeuronLayer,Neuron
import math
from copy import copy
import sys
import numpy as np
import matplotlib.pyplot as plt

# A class to store a 20Q round
# Takes in a target (correct answer) and nueral network
# Autoplay: generates an input vector for training / testing the network
# Play: interactive game between user and bot

class Q20(object):

	def __init__(self,nn,qs,ts,q_limit):

		self.qs = qs
		self.ts = ts
		self.q_limit = q_limit # q_limit <= num_qs

		self.num_qs = len(qs)
		self.num_ts = len(ts)

		self.nn = nn # brain

	# randomly generate set of questions + answers for target
	def autoplay(self,target):

		# initialise input vector
		input_vector = [0]*self.num_qs
		qs_asked = []

		# prepare target vector based on supplied target
		target_vector = [0]*self.num_ts
		target_vector[target] = 1

		# randomly generate q_limit questions
		for __ in range(self.q_limit):

			# select a random question that hasn't been asked
			qi = random.randint(0,self.num_qs-1)
			while qi in qs_asked:
				qi = random.randint(0,self.num_qs-1)

			input_vector[qi] = Z.get_answer_i(qi,target) # returns 1 for yes, -1 for no

			qs_asked.append(qi)

		# train the network object
		self.nn.backpropagate(input_vector,target_vector)

		return input_vector

	# autoplay with smart question selection
	def autoplay_smartq(self,target):

		output = [0]*self.num_ts # initialise output for first q selection
		input_vector = [0]*self.num_qs

		target_vector = [0]*self.num_ts
		target_vector[target] = 1

		qi_order = []
		o_order = []

		qs_asked = []

		# guessed = []
		best_guess = None

		for i in range(self.q_limit):

			# index of the best guess at this point in the game
			best_guess = output.index(max(output))

			# get best next question
			best_qi = self.smart_q(output,input_vector,qs_asked)
			if best_qi == -1: break # no more questions

			# set the question index to +1 or -1 depending on the response
			input_vector[best_qi] = Z.get_answer_i(best_qi,target)
			if best_qi not in qs_asked: qs_asked.append(best_qi)
			qi_order.append(best_qi)

			# feed the network forward with the new input vector
			output = self.nn.feed_forward(input_vector)

			o_order.append([self.ts[best_guess],100*round(max(output),2)])

		# set up q/a string for print
		qs_str = []
		for i in range(len(qi_order)):
			ans = Z.get_answer_i(qi_order[i],target)
			if ans == 1: ansstr = ' y '
			else: ansstr = ' n '
			q_str = str(i+1)+'. '+qs[qi_order[i]] + ansstr + '(' + str(o_order[i][0]) + ' ' + str(o_order[i][1]) + '%)'
			qs_str.append(q_str)

		if best_guess == target:
			# reinforce successful win
			print 'win:',self.ts[best_guess],'...',len(qs_str),'qs:',' '.join(qs_str)
			return (1,input_vector)
		else:
			print 'LOSS:',self.ts[best_guess],'(',self.ts[target],')','...',len(qs_str),'qs:',' '.join(qs_str)
			return (0,input_vector)

	# smart question selection algorithm based on the current probabilities (output) and questions asked (input_vector, qs_asked)
	def smart_q(self,output,input_vector,qs_asked):

		best_qi = -1
		best_diff = -1

		for j in range(self.num_qs):

			if j in qs_asked: continue # question has already been asked, ignore

	    	# test vectors for q-yes and q-no
			test_vector_no = copy(input_vector)
			test_vector_no[j] = -1
			test_vector_yes = copy(input_vector)
			test_vector_yes[j] = 1

	    	# test outputs for q-yes
			test_output_yes = self.nn.feed_forward(test_vector_yes)

			# test outputs for q-no
			test_output_no = self.nn.feed_forward(test_vector_no)

	    	# absolute differences between current probabilities and hypothetical probabilities
			test_diff_yes = sum([abs(x-y) for x,y in zip(test_output_yes,output)])
			test_diff_no = sum([abs(x-y) for x,y in zip(test_output_no,output)])

			# worse case diff score
			if test_diff_yes > test_diff_no: test_diff = test_diff_no
			else: test_diff = test_diff_yes

	    	# select the question with the greatest diff,
	    	# i.e. most descriptive jump from current probabilities,
	    	# of the worst case answer for each question
			if best_qi == -1 or test_diff >= best_diff:
				best_diff = test_diff
				best_qi = j

		return best_qi

	# user mode
	def play(self):

		print '\n================ YOUR TURN: SELECT ANIMAL ================'
		print 'select an animal from the list:'
		print '[',', '.join(self.ts),']'

		raw_input("press enter to start")

		print '====================== '+str(self.q_limit)+' QUESTIONS ======================'

		# play game
		output = [0]*self.num_ts # initialise output for first q selection
		input_vector = [0]*self.num_qs
		qs_asked = []

		for i in range(self.q_limit-1):

			# index of the best guess at this point in the game
			best_guess = output.index(max(output))

			# get the next best question
			best_qi = self.smart_q(output,input_vector,qs_asked)
			if best_qi == -1: break # we are out of questions
			print str(i+1)+'. '+self.qs[best_qi]
			if best_qi not in qs_asked: qs_asked.append(best_qi)

			# get user input as answer
			ans_str = raw_input('Answer? (y/n) ')
			while ans_str != 'y' and ans_str != 'n':
				ans_str = raw_input('Invalid. Answer? (y/n) ')

			if ans_str == 'y': ans = 1
			else: ans = -1

			# add the question response to the input vector
			input_vector[best_qi] = ans
			output = self.nn.feed_forward(input_vector)

			print '('+self.ts[best_guess]+' '+str(100*round(max(output),2))+'%)'

		# best final guess
		best_guess = output.index(max(output))

		# initialise target vector
		target_vector = [0]*self.num_ts

		print self.q_limit,'. are you thinking of a',self.ts[best_guess],'?'
		ans_str = raw_input('Answer? (y/n) ')
		while ans_str != 'y' and ans_str != 'n':
			ans_str = raw_input('Invalid. Answer? (y/n) ')
		if ans_str == 'y':
			print '\nBOT WINS.'
			# fill target vector with correct guess
			target_vector[best_guess] = 1
		else:
			print '\nYOU WIN.\n'
			print 'what animal were you thinking of?'
			t_str = raw_input('Animal: ')
			while t_str not in self.ts:
				t_str = raw_input('Not in list, select again. Animal: ')
			# request the correct animal for training
			target_vector[self.ts.index(t_str)] = 1

		# train the network based on the win / loss
		print 'training network...'
		self.nn.backpropagate(input_vector,target_vector)


# # # # # # #
# TRAINING  # 
# # # # # # #

def train(nn,training_size):

	i = 0
	interval = int(training_size/100)

	game = Q20(nn,qs,ts,question_limit)

	for __ in range(int(training_size/len(targets))):

		t = 0
		for target_vector in targets:

			input_vector = game.autoplay(t)

			i+=1
			t+=1

	return nn

# # # # # # # #
# VALIDATING  # 
# # # # # # # #

def validate(nn,num_games):

	game = Q20(nn,qs,ts,question_limit)

	print '\n======================================='
	print 'VALIDATION SET'
	print '=======================================\n'

	wins = 0

	testing_sets = []

	for t in range(num_games):

		t = random.randint(0,num_ts-1)
		target_vector = targets[t]
		win,input_vector = game.autoplay_smartq(t)
		testing_sets.append([input_vector,target_vector])

		if win == 1: wins+=1

	accuracy = (float(wins)/float(num_games))
	error = nn.calculate_total_error(testing_sets)

	print '======================================='
	print accuracy*100,'% accuracy, error=',error
	print '======================================='

	return (error,accuracy)

# # # # # #
# TESTING # 
# # # # # #

def test(nn):
	game = Q20(nn,qs,ts,question_limit)
	while True:
		game.play()
		nn.save('saved/'+SRC+'.json')


# cross-validate process (training a network)

def crossvalidate():

	learning_rate = 0.1

	if SRC == 'micro': hidden_size = 4
	elif SRC == 'medium': hidden_size = 12
	elif SRC == 'small': hidden_size = 8
	else: hidden_size = 38

	num_weights = int(hidden_size*input_size + output_size*hidden_size)

	nn = NeuralNetwork(num_inputs=input_size,num_hidden=hidden_size,num_outputs=output_size,learning_rate=learning_rate,hidden_layer_bias=1,output_layer_bias=1)

	print '\n======================================='
	print 'NETWORK SUMMARY'
	print '=======================================\n' 

	print '#input =',input_size
	print '#output =',output_size
	print '#hidden =',hidden_size
	print 'learning rate =',learning_rate
	print '#weights =',num_weights

	# setup epoch intervals, upper epoch = #weights^2
	upper_epoch = round(num_weights*num_weights,100)
	if upper_epoch > 1000000: upper_epoch = 1000000 # max 1m epoch
	interval = int(upper_epoch/10) # 10 intervals
	validate_epoch = int(interval/10)
	if validate_epoch > 200: validate_epoch = 200 # max 200 validation set (to save computational time)

	print 'upper epoch =',upper_epoch
	print 'interval =',interval
	print '#validation sets =',validate_epoch
	print 'error breakpoint <=',validate_epoch/10

	i = 0

	epoch = 0

	epoch_intervals = []
	epoch_errors = []
	epoch_accuracy = []

	while epoch < upper_epoch:

		epoch += interval

		print '\n======================================='
		print i,'. # TRAINING SETS:',epoch
		print '=======================================\n' 

		train(nn,interval)
		error,accuracy = validate(nn,validate_epoch)

		epoch_intervals.append(epoch)
		epoch_errors.append(error)
		epoch_accuracy.append(accuracy)

		if error < validate_epoch/10: break # once MSE is small enough, break

		i+=1

	# save network temporarily
	nn.save('saved/'+SRC+'-tmp.json')

	# plot epoch vs error, accuracy
	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()
	ax1.plot(epoch_intervals,epoch_errors,'b-o')
	ax2.plot(epoch_intervals,epoch_accuracy,'r-o')

	plt.title('Epoch vs error\nq_limit = '+str(question_limit)+' ('+str(SRC)+')')
	ax1.set_ylabel('Error')
	ax2.set_ylabel('Accuracy')
	ax1.set_xlabel('Epoch (#training sets)')
	plt.show()

	test(nn) # play

def test_learning_rate():

	learning_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
	training_errors = []
	training_accuracy = []

	# hardcoded for 'big'

	epoch = 150000
	hidden_size = 38

	i = 0
	for curr_rate in learning_rates:

		print '\n======================================='
		print i+1,'. LEARNING RATE:',curr_rate
		print '=======================================\n' 

		nn = NeuralNetwork(num_inputs=input_size,num_hidden=hidden_size,num_outputs=output_size,learning_rate=curr_rate,hidden_layer_bias=1,output_layer_bias=1)

		train(nn,epoch)
		error,accuracy = validate(nn,1000)

		training_errors.append(error)
		training_accuracy.append(accuracy)

		i+=1

	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()
	ax1.plot(learning_rates,training_errors,'b-o')
	ax2.plot(learning_rates,training_accuracy,'r-o')

	plt.title('Learning rate vs error\n('+str(SRC)+', epoch = '+str(epoch)+')')
	ax1.set_ylabel('Error')
	ax2.set_ylabel('Accuracy')
	ax1.set_xlabel('Learning rate')
	plt.show()

def test_num_hidden():

	# hardcoded for 'big'

	epoch = 150000

	training_hidden = []
	training_errors = []
	training_accuracy = []

	i = 0
	for x in range(10,50,2):

		print '\n======================================='
		print i+1,'. # HIDDEN:',x
		print '=======================================\n' 

		nn = NeuralNetwork(num_inputs=input_size,num_hidden=x,num_outputs=output_size,learning_rate=0.8,hidden_layer_bias=1,output_layer_bias=1)

		train(nn,epoch)
		error,accuracy = validate(nn,1000)

		training_errors.append(error)
		training_hidden.append(x)
		training_accuracy.append(accuracy)

		i+=1

	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()
	ax1.plot(training_hidden,training_errors,'b-o')
	ax2.plot(training_hidden,training_accuracy,'r-o')

	plt.title('# hidden neurons vs error\n('+str(SRC)+', epoch = '+str(epoch)+')')
	ax1.set_ylabel('Error')
	ax2.set_ylabel('Accuracy')
	ax1.set_xlabel('# hidden neurons')
	plt.show()


def load():

	# load network and play game
	filename = 'saved/'+SRC+'.json'

	print '\nloading network from',filename
	nn = NeuralNetwork(loadfile=filename) # load network

	print '\n======================================='
	print 'NETWORK SUMMARY'
	print '=======================================\n' 

	print '#input =',nn.num_inputs
	print '#output =',nn.num_outputs
	print '#hidden =',nn.num_hidden

	# validate(nn,50) # validate loaded network

	test(nn) # play


# global variables
question_limit = 0
qs = []
num_qs = 0
ts = []
num_ts = 0
input_size = 0
output_size = 0
learning_rate = 0.0

if __name__ == "__main__":

	# initialise global variables

	SRC = 'big' # default source is big dataset

	if len(sys.argv) >= 2: SRC = sys.argv[1] # take user source

	# get questions and answers from the Zoo class
	Z = Zoo('data/'+SRC+'.csv')
	qs = Z.questions
	num_qs = len(qs)
	ts = Z.targets
	num_ts = len(ts)

	# hardcoded question limits for each dataset (+1 for last question guess)
	if SRC == 'big': question_limit = 13+1
	elif SRC == 'medium': question_limit = 6+1
	elif SRC == 'small': question_limit = 5+1
	elif SRC == 'micro': question_limit = 4+1
	else: question_limit = len(qs)

	# input and output size for nn are equal to the num qs and num ts respectively
	input_size = len(qs)
	output_size = len(ts)

	# list of target vectors form a diagonal matrix
	targets = [[1 if i == j else 0 for i in range(output_size)]
	               for j in range(output_size)]

	# default mode is play (load from existing network)
	MODE = 'play'
	if len(sys.argv) >= 3: MODE = sys.argv[2]

	if MODE == 'play': load()
	elif MODE == 'crossvalidate': crossvalidate()
	# elif MODE == 'hidden': test_num_hidden()
	# elif MODE == 'learningrate': test_learning_rate()


