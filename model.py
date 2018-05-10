import sys
import torch as tc
import numpy as np
import random
import time

class Flatten(tc.nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		n = x.shape[0]
		m = x.shape[1]
		x = x.view(n, m)
		return x

def weights_init(m):
    if isinstance(m, tc.nn.Conv3d):
        tc.nn.init.xavier_normal_(m.weight)
        m.bias.data.zero_()
    return

def cnn_module():
	main = tc.nn.Sequential()
	# 65 * 65 * 65 , 1
	main.add_module('avg_pool_1', tc.nn.AvgPool3d((2, 2, 2), stride=(2, 1, 1)))
	# 32 * 64 * 64 , 1
	main.add_module('conv_3d_1', tc.nn.Conv3d(1, 32, 3, 1, 1))
	main.add_module('relu_1', tc.nn.ReLU())
	# 32 * 64 * 64 , 32
	main.add_module('max_pool_1', tc.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)))
	# 32 * 32 * 32 , 32
	main.add_module('conv_3d_2', tc.nn.Conv3d(32, 64, 3, 1, 1))
	main.add_module('relu_2', tc.nn.ReLU())
	# 32 * 32 * 32 , 64
	main.add_module('max_pool_2', tc.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)))
	# 16 * 16 * 16 , 64
	main.add_module('conv_3d_3', tc.nn.Conv3d(64, 128, 3, 1, 1))
	main.add_module('relu_3', tc.nn.ReLU())
	# 16 * 16 * 16 , 128
	main.add_module('max_pool_3', tc.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)))
	# 8 * 8 * 8 , 128
	main.add_module('conv_3d_4', tc.nn.Conv3d(128, 256, 3, 1, 1))
	main.add_module('relu_4', tc.nn.ReLU())
	# 8 * 8 * 8 , 256
	main.add_module('max_pool_4', tc.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)))
	# 4 * 4 * 4, 256
	main.add_module('conv_3d_5', tc.nn.Conv3d(256, 512, 3, 1, 1))
	main.add_module('relu_5', tc.nn.ReLU())
	# 4 * 4 * 4 , 512
	main.add_module('max_pool_5', tc.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)))
	# 2 * 2 * 2 , 512
	main.add_module('conv_3d_6', tc.nn.Conv3d(512, 64, 2, 1, 0))
	main.add_module('relu_6', tc.nn.ReLU())
	# 1 * 1 * 1 , 64
	main.add_module('conv_3d_7', tc.nn.Conv3d(64, 1, 1, 1, 0))
	main.add_module('sigmoid_1', tc.nn.Sigmoid())
	# 1 * 1 * 1 , 1
	main.add_module('flatten_1', Flatten())

	main.add_module('conv_3d_8', tc.nn.Conv3d(64, 1, 1, 1, 0))
	main.add_module('flatten_2', Flatten())
	return main

class Net(tc.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.layers = cnn_module()

	def forward(self, x, train):
		for i in range(0, 18):
			x = self.layers[i](x)
		x1 = self.layers[20](self.layers[19](self.layers[18](x)))
		x2 = self.layers[22](self.layers[21](x))
		return tc.cat((x1, x2), 1)

def load_data():
	x1 = np.load('./data/cubes-0.npz')['cubes']
	y1 = np.load('./data/labels-0.npz')['labels']
	x2 = np.load('./data/cubes-1.npz')['cubes']
	y2 = np.load('./data/labels-1.npz')['labels']
	pos = np.load('./data/positives.npy')
	pos_y = np.array([1] * len(pos))
	cut1_1 = int(len(x1)*0.1)
	cut2_1 = int(len(x2)*0.1)
	cut3_1 = int(len(pos)*0.1)
	cut1_2 = int(len(x1)*0.3)
	cut2_2 = int(len(x2)*0.3)
	cut3_2 = int(len(pos)*0.3)
	# xtrain = np.concatenate((x1[:cut1_1], x2[:cut2_1], pos[:cut3_1]), axis=0)
	# xlabel = np.concatenate((y1[:cut1_1], y2[:cut2_1], pos_y[:cut3_1]), axis=0)
	# xvalid = np.concatenate((x1[cut1_1:cut1_2], x2[cut2_1:cut2_2], pos[cut3_1:cut3_2]), axis=0)
	# validlabel = np.concatenate((y1[cut1_1:cut1_2], y2[cut2_1:cut2_2], pos_y[cut3_1:cut3_2]), axis=0)
	# xtest = np.concatenate((x1[cut1_2:], x2[cut2_2:], pos[cut3_2:]), axis=0)
	# testlabel = np.concatenate((y1[cut1_2:], y2[cut2_2:], pos_y[cut3_2:]), axis=0)
	xtest = np.concatenate((x1[:cut1_1], x2[:cut2_1], pos[:cut3_1]), axis=0)
	testlabel = np.concatenate((y1[:cut1_1], y2[:cut2_1], pos_y[:cut3_1]), axis=0)
	xvalid = np.concatenate((x1[cut1_1:cut1_2], x2[cut2_1:cut2_2], pos[cut3_1:cut3_2]), axis=0)
	validlabel = np.concatenate((y1[cut1_1:cut1_2], y2[cut2_1:cut2_2], pos_y[cut3_1:cut3_2]), axis=0)
	xtrain = np.concatenate((x1[cut1_2:], x2[cut2_2:], pos[cut3_2:]), axis=0)
	xlabel = np.concatenate((y1[cut1_2:], y2[cut2_2:], pos_y[cut3_2:]), axis=0)
	return xtrain, xlabel, xvalid, validlabel, xtest, testlabel

# def load_test_data():
# 	x = np.load('./data/cubes-1.npz')['cubes']
# 	y = np.load('./data/labels-1.npz')['labels']
# 	return x, y

def create_batch(x, y, batch_size):
	num_instance = [i for i in range(len(x))]
	r = random.sample(num_instance, batch_size)
	batch_x = np.empty(0)
	batch_y = np.empty(0)
	for i in r:
		batch_x = np.append(batch_x, x[i])
		batch_y = np.append(batch_y, y[i])
	return np.reshape(batch_x, (int(batch_size),1,65,65,65)), batch_y

def main(params):
	print('-'*80)
	print('loading data')
	xtrain, xlabel, xvalid, validlabel, xtest, testlabel = load_data()
	# xtrain, xlabel = load_train_data()
	# xtest, testlabel = load_test_data()

	model = Net().cuda()
	# model.load_state_dict(tc.load('./checkpoint.pt'))
	model.apply(weights_init)
	model.train()

	optimizer = tc.optim.SGD(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'], momentum=params['momentum'])

	print('-'*80)
	print('start training')
	for epoch in range(1, params['train_epoch']+1):
		print('-'*80)
		start = time.time()
		count = 0
		train_loss = []
		for i in range(int(len(xtrain)/params['batch_size'])):
			data, label = create_batch(xtrain, xlabel, params['batch_size'])
			data, label = tc.autograd.Variable(tc.FloatTensor(data), requires_grad=True).cuda(), tc.autograd.Variable(tc.FloatTensor(label)).cuda()
			optimizer.zero_grad()
			output = model(data, train=True).cuda()
			output_label = output.data.max(1, keepdim=True)[1].cpu().numpy()
			index = 0
			for y in output_label:
				if y[0] != label[index]:
					count += 1
				index += 1
			loss = tc.nn.functional.cross_entropy(output.float(), label.long())
			loss.backward()
			optimizer.step()
			print('Batch: ' + str(i) + ' Loss: ' + str(loss.item()))
			train_loss.append(loss.item())
		print('epoch ' + str(epoch) + ' finishes in ' + str(time.time() - start))
		print('-'*80)
		error_rate = count / len(xlabel)
		print('train error rate: ' + str(error_rate))
		print('train cross cross_entropy loss： ' + str(sum(train_loss) / len(train_loss)))

		min_error_rate = 1

		if epoch % 1 == 0:
			print('-'*80)
			count = 0
			for i in range(int(len(xvalid)/params['batch_size'])):
				data, label = create_batch(xvalid, validlabel, params['batch_size'])
				data = tc.autograd.Variable(tc.FloatTensor(data), volatile=True).cuda()
				output = model(data, train=False).cuda().data.max(1, keepdim=True)[1].cpu().numpy()
				index = 0
				for y in output:
					if y[0] != label[index]:
						count += 1
					index += 1
			error_rate = count/len(validlabel)
			if error_rate < min_error_rate:
				tc.save(model.state_dict(), './checkpoint.pt')
				min_error_rate = error_rate
			print('validation error rate: ' + str(error_rate))

	print('-'*80)
	model.eval()
	count = 0
	for i in range(int(len(xtest)/params['batch_size'])):
		data, label = create_batch(xtest, testlabel, params['batch_size'])
		data = tc.autograd.Variable(tc.FloatTensor(data), volatile=True).cuda()
		output = model(data, train=False).cuda().data.max(1, keepdim=True)[1].cpu().numpy()
		index = 0
		for y in output:
			if y[0] != label[index]:
				count += 1
			index += 1
	print('error rate: ' + str(count/len(testlabel)))


if __name__ == '__main__':
	params = {}
	params['train_epoch'] = 30
	params['batch_size'] = 32
	params['learning_rate'] = 0.01
	params['momentum'] = 0.9
	params['weight_decay'] = 0.0001
	main(params)
