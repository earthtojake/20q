from zoo2 import Zoo
import random
from neural import NeuralNetwork,NeuronLayer,Neuron
import math
from copy import copy
import progressbar
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
		self.q_limit = q_limit

		self.num_qs = len(qs)
		self.num_ts = len(ts)

		self.nn = nn # brain

	# randomly generate set of questions + answers for target
	def autoplay(self,target):

		input_vector = [0]*self.num_qs*2
		qs_asked = []

		target_vector = [0]*self.num_ts
		target_vector[target] = 1

		for __ in range(self.q_limit):

			qi = random.randint(0,num_qs-1)
			while qi in qs_asked: # wait for a question that hasn't been asked
				qi = random.randint(0,num_qs-1) # select random question from list

			ans = Z.get_answer_i(qi,target) # returns 1 for yes, 0 for no
			qii = 2*qi + ans # question index of q-no or q-yes answer
			input_vector[qii] = 1
			qs_asked.append(qi)

		self.nn.train(input_vector,target_vector)

		return input_vector

	# autoplay with smart question selection
	def autoplay_smartq(self,target):

		output = [0]*output_size # initialise output for first q selection
		input_vector = [0]*input_size

		target_vector = [0]*self.num_ts
		target_vector[target] = 1

		qi_order = []
		o_order = []

		qs_asked = []

		for __ in range(self.q_limit):

			best_qi = self.next_best_q(output,input_vector,qs_asked)

			if best_qi == -1: break

			ans = Z.get_answer_i(best_qi,target) # returns 1 for yes, 0 for no
			qii = 2*best_qi + ans # question index of q-no or q-yes answer
			input_vector[qii] = 1
			if best_qi not in qs_asked: qs_asked.append(best_qi)

			qi_order.append(best_qi)

			output = self.nn.feed_forward(input_vector)

			o_order.append([self.ts[output.index(max(output))],round(max(output),2)])

		guess_t = output.index(max(output))

		qs_str = []
		for i in range(len(qi_order)):
			q_str = qs[qi_order[i]] + ' (' + str(o_order[i][0]) + ' ' + str(o_order[i][1]) + ')'
			qs_str.append(q_str)

		if guess_t == target:
			# reinforce successful win
			print 'win:',self.ts[guess_t],'...',len(qs_str),'qs:',' '.join(qs_str)
			return (1,input_vector)
		else:
			print 'LOSS:',self.ts[guess_t],'(',self.ts[target],')','...',len(qs_str),'qs:',' '.join(qs_str)
			return (0,input_vector)

	def next_best_q(self,output,input_vector,qs_asked):

		best_qi = -1
		best_diff = -1

		for j in range(self.num_qs):

			if j in qs_asked: continue # question has already been asked, ignore

	    	# test vectors for q-yes and q-no
			test_vector_no = copy(input_vector)
			test_vector_no[2*j] = 1
			test_vector_yes = copy(input_vector)
			test_vector_yes[2*j+1] = 1

	    	# test outputs for q-yes
			test_output_yes = self.nn.feed_forward(test_vector_yes)

			# test outputs for q-no
			test_output_no = self.nn.feed_forward(test_vector_no)

	    	# average of test outputs (assuming 50/50 chance of a yes-no q)
			test_diff_yes = sum([abs(x-y) for x,y in zip(test_output_yes,output)])
			test_diff_no = sum([abs(x-y) for x,y in zip(test_output_no,output)])

			if test_diff_yes > test_diff_no: test_diff = test_diff_no
			else: test_diff = test_diff_yes

	    	# select the biggest standard deviation, i.e. biggest separation of output probabilities,
	    	# for the worst case answer in each question
			if best_qi == -1 or test_diff >= best_diff:
				best_diff = test_diff
				best_qi = j

		return best_qi

	def play(self):

		print '\n================ YOUR TURN: SELECT ANIMAL ================'
		print '~Your animal will not be shared with the bot until it makes its final guess'

		target = raw_input("Your animal is: ")
		while target not in ts:
			print 'Animal not in list. Select another...'
			target = raw_input("Your animal is: ")

		t = self.ts.index(target)
		target_vector = [0]*self.num_ts
		target_vector[t] = 1

		print '====================== '+str(self.q_limit)+' QUESTIONS ======================'

		# play game
		output = [0]*output_size # initialise output for first q selection
		input_vector = [0]*input_size
		qs_asked = []

		for i in range(self.q_limit):

			best_qi = self.next_best_q(output,input_vector,qs_asked)
			if best_qi == -1: break

			print str(i+1)+'. '+self.qs[best_qi]
			ans_str = raw_input('Answer? (y/n) ')
			while ans_str != 'y' and ans_str != 'n':
				ans_str = raw_input('Invalid. Answer? (y/n) ')
			if ans_str == 'y': ans = 1
			else: ans = 0

			qii = 2*best_qi + ans # question index of q-no or q-yes answer
			input_vector[qii] = 1

			if best_qi not in qs_asked: qs_asked.append(best_qi)

			output = self.nn.feed_forward(input_vector)

			print '(',self.ts[output.index(max(output))],100*round(max(output),2),'% )'

		guess_t = output.index(max(output))

		if guess_t == t:
			print 'Bot guessed \'',target,'\' correctly!'
		else:
			print 'Bot guessed \'',self.ts[guess_t],'\' incorrectly! (',target,')'

		print 'Training network...'
		self.nn.train(input_vector,target_vector)


def train_test(nn,training_size):

	bar = progressbar.ProgressBar(maxval=training_size,widgets=[progressbar.Bar('=','[',']'),' ',progressbar.Percentage()])
	bar.start()

	i = 0
	interval = int(training_size/100)

	game = Q20(nn,qs,ts,question_limit)

	for __ in range(int(training_size/len(targets))):

		t = 0
		for target_vector in targets:

			input_vector = game.autoplay(t)

			try:
				if i % interval == 0:bar.update(i+interval)
			except:
				pass

			i+=1
			t+=1

	bar.finish()

	print '\n======================================='
	print 'VALIDATION SET'
	print '=======================================\n'

	wins = 0
	num_games = 100

	testing_sets = []

	for i in range(num_games):

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

# global variables

if len(sys.argv) >= 2: SRC = sys.argv[1]
else: SRC = 'zoo_medium1.csv' 

Z = Zoo(SRC)

qs = Z.questions
num_qs = len(qs)
ts = Z.targets
num_ts = len(ts)

if len(sys.argv) >= 3 and int(sys.argv[2]) <= len(qs): question_limit = int(sys.argv[2])
else: question_limit = len(qs)

input_size = len(qs)*2
output_size = len(ts)
hidden_size = int(math.sqrt(input_size+output_size))+2
training_case_size = question_limit

learning_rate = 0.8

num_weights = int(hidden_size*input_size + output_size*hidden_size)

targets = [[1 if i == j else 0 for i in range(output_size)]
               for j in range(output_size)]

upper_epoch = int(num_weights*num_weights)

def crossvalidate():

	# medium2 settings: 12 hidden neurons, 0.8 learning rate, upper epoch = 20k
	nn = NeuralNetwork(num_inputs=input_size,num_hidden=23,num_outputs=output_size,learning_rate=0.8,hidden_layer_bias=1,output_layer_bias=1)

	print '\n======================================='
	print 'NETWORK SUMMARY'
	print '=======================================\n' 

	print 'src =',SRC
	print 'input =',input_size
	print 'output =',output_size
	print 'hidden =',23
	print 'question limit =',question_limit
	print 'learning rate =',learning_rate
	print 'num weights =',num_weights

	interval = 100000
	i = 0

	epoch = 0

	epoch_intervals = []
	epoch_errors = []

	while True:

		epoch += interval

		print '\n======================================='
		print i,'TRAINING SETS:',epoch
		print '=======================================\n' 

		error,accuracy = train_test(nn,interval)

		epoch_intervals.append(epoch)
		epoch_errors.append(error)

		if error < 10 or accuracy >= 0.95: break

		i+=1

	nn.inspect()

	nn.save(SRC[:-4]+'.json')

	plt.title('Epoch vs error\nq_limit = '+str(question_limit)+' ('+str(SRC)+')')
	plt.ylabel('Error')
	plt.xlabel('Epoch (#training sets)')
	plt.plot(epoch_intervals,epoch_errors,'-o')
	plt.show()

	# now play the game...
	game = Q20(nn,qs,ts,question_limit)
	while True:
		game.play()

def test_learning_rate():

	learning_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
	training_errors = []

	epoch = 300000

	i = 0
	for curr_rate in learning_rates:

		print '\n======================================='
		print i,'LEARNING RATE:',curr_rate
		print '=======================================\n' 

		nn = NeuralNetwork(num_inputs=input_size,num_hidden=18,num_outputs=output_size,learning_rate=curr_rate,hidden_layer_bias=1,output_layer_bias=1)

		error,accuracy = train_test(nn,epoch)

		training_errors.append(error)

		i+=1

	plt.title('Learning rate vs error\n('+str(SRC)+', epoch = '+str(epoch)+')')
	plt.ylabel('Error')
	plt.xlabel('Learning rate')
	plt.plot(learning_rates,training_errors,'-o')
	plt.show()

def test_num_hidden():

	epoch = 500000

	training_hidden = []
	training_errors = []

	# medium2 = 40000

	i = 0
	for x in range(45,80,5):

		print '\n======================================='
		print i,'NUMBER HIDDEN:',x
		print '=======================================\n' 

		nn = NeuralNetwork(num_inputs=input_size,num_hidden=x,num_outputs=output_size,learning_rate=0.8,hidden_layer_bias=1,output_layer_bias=1)

		error,accuracy = train_test(nn,epoch)

		training_errors.append(error)
		training_hidden.append(x)

		# if error > 500: break

		i+=1

	plt.title('# hidden neurons vs error\n('+str(SRC)+', epoch = '+str(epoch)+')')
	plt.ylabel('Error')
	plt.xlabel('# hidden neurons')
	plt.plot(training_hidden,training_errors,'-o')
	plt.show()


if __name__ == "__main__":
	# test_learning_rate()
	# test_num_hidden()
	crossvalidate()