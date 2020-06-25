#language-model

import torchtext
from torchtext.vocab import Vectors
import  torch
import numpy as np 
import random

USE_CUDA = torch.cuda.is_available()

random.seed(9567)
np.random.seed(9567)
torch.manual_seed(9567)
if USE_CUDA:
	torch.cuda.manual_seed(9567)

BATCH_SIZE = 32
EMBEDDING_SIZE = 128
HIDDEN_SIZE  =  100
MAX_VOCAB_SIZE = 10000

#定义一个field,声明输入的是什么
TEXT = torchtext.data.Field(lower = True)
train,val,test = torchtext.datasets.LanguageModelingDataset.splits( path = '.',train = 'text8.train.txt',validation = 'text8.dev.txt',test = 'text8.test.txt',text_field = TEXT) 

TEXT.build_vocab(train,max_size = MAX_VOCAB_SIZE)
device = torch.device('cuda' if USE_CUDA else 'cpu')
VOCAB_SIZE = len(TEXT.vocab)

train_iter,val_iter,test_iter = torchtext.data.BPTTIterator.splits(train,val,test),batch_size = BATCH_SIZE,device = device,bptt_len=16,repeat = False,shuffle = True

#定义模型
import torch.nn as nn
class RNNModel(nn.Module):
	def __init__(self,vocab_size,embed_size,hidden_size):
		super(RNNModel,self).__init__()
		self.encoder = nn.Embedding(vocab_size,embed_size)
		self.lstm = nn.LSTM(embed_size,hidden_size)
		self.linear = nn,Linear(hiddden_size,vocab_size)
		self.hidden_size  = hidden_size

	def forward(self,text,hidden):
		#text:seq_length * batch_size
		emd = self.encoder(text)#seq_length*batch_size*embed_size
		ouput,hidden = self.lstm(emb,hidden)#output:seq_len*batch_size* hidden_size,hidden:(1*batch_size* hidden_size,1*batch_size* hidden_size)
		#把output的前两个维度拼到一起
        #(seq_len*batch_size)* hidden_size
		out_vocab = self.linear(ouput.view(-1,output.shape[2]))
		out_vocab = out_vocab.view(output.size(0),output.size(1),out_vocab.size(-1))
		return out_vocab,hidden
	def ini_hidden(self,bas,requires_grad = True):
		weight = next(self,parameters())
		return (weight.new_zeros((1,bsz,self.hidden_size),requires_grad = True),weight.new_zeros((1,bsz,self.hidden_size),requires_grad = True))
model = RNNModel(vocab_size - len(TEXT.vocab),embed_size = EMBEDDING_SIZE,hidden_size = HIDDEN_SIZE)

#训练模型

def repackage_hidden(h):
	if isinstance(h,torch.Tensor):
		return h.detach()
	else:
		return tuple(repackage_hidden(v) for v in h)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(),
                            lr = learning_rate)

def evaluate(model,data):
	model.eval()
	total_loss = 0.
	total_count = 0.
	it = iter(data)
	with torch.no_grad():
		hidden = model.init_hidden(BATCH_SIZE)
		for i,batch in enumerate(it):
			data,target = batch.text,batch.target
			hidden = repackage_hidden(hidden)
			output,hidden = model(data,hidden)
			loss = loss_fn(output.view(-1,VOCAB_SIZE),target.view(-1))
			total_loss+= loss.item()*np.multiply(*data.size())
			total_count+=np.multiply(*data.size())
	loss = total_loss
	model.train()
	return loss

NUM_EPOCHS = 2
VOCAB_SIZE = len(TEXT.vocab)
GRAD_CLIP = 5
for epoch in range(NUM_EPOCHS):
	model.train()
	it = iter(train_iter)
	hidden = model,init_hidden(BATCH_SIZE)
	for i,batch in enumerate(it):
		data,target = batch.text,batch.target
		hidden = repackage_hidden(hidden)
		output,hidden = model(data,hidden)

		loss = loss_fn(output.view(-1,VOCAB_SIZE),target.view(-1))
		optimizer.zeros_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
		optimizer.step()




