# define the path to the input.txt file
file_path = 'Datasets\\shakespeare.txt'

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# print("Length of dataset in characters: ", len(text))

# examine the first 1000 characters
# print(text[:1000])

# examine all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars) # possible elements of our sequences
print(''.join(chars)) 
print(vocab_size)

# strategy to tokenize input text
# tokenize means to convert the raw string (as text) to some sequence of integers according to some notebook
# because this is a character level model, we will translate individual characters into integers
'''
# create a mapping from characters to integers (character level tokenizer)
# this is one of many possible encodings/tokenizers (a very simple one)
# Google uses SentencePiece (https://github.com/google/sentencepiece), which encodes text into integers, but in a different schema and using a different vocabulary. It is a sub-word tokenizer, a middle ground between character encoding and word encoding.
# OpenAI & GPT use tiktoken (https://github.com/openai/tiktoken), which is a byte pair encoding tokenizer. This enables encoding words into integers. Instead of having 65 possible characters/tokens, they have 50,000+ tokens.
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# when we encode an arbitrary text (like "hi there"), we are going to receive a list of integers that represents that string
# we can take a list of integers and decode it to get back the exact same string
# this process can be seen as a translation of characters to integers (and back)
# print(encode("hii there"))
# print(decode(encode("hii there")))

# encode the entire text dataset and store it into a torch.Tensor
# a tensor is a multi-dimensional array of data (1 cell, 1 array, 1 matrix, 1 "cube" of values, etc.)
import torch # we use PyTorch: https://pytorch.org
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000]) # the 1000 characters we looked at earlier will (to the GPT) look like this

# split up the data into a training set and a validation set
n = int(0.9*len(data)) # first 90% will be for training, the rest for validation
train_data = data[:n]
val_data = data[n:]

# feeding an entire dataset to a transformer all at once would be computationally very expensive and prohibitive.
# when we train a transformer on a dataset, we only work with chunks of the dataset. When we train the transformer, we (basically) sample random chunks out of the training set and train them just chunks at a time.
# The chunks have a maximum length. That maximum length is called block size, context length, etc.
block_size = 8
train_data[:block_size+1]
# print(train_data)

# When you sample a chunk of data like train_data (the nine characters out of the training set), train_data has multiple examples in it. All of the characters in train_data "follow each other"
# When we plug train_data into a transformer, we are going to simultaneously train it to make a prediction at every one of the positions. 
# In a chunk of nine characters, there's eight individual examples packed into it. An "example" is an integer, and the integer you expect to come next. 
# Here is code to illustrate:
x = train_data[:block_size] # inputs to the transformer
y = train_data[1:block_size+1] # targets for each position in the input
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    # print(f"when input is {context} the target: {target}")

# we train on all eight examples with context between one all the way up to block_size. 
# we train on that not just for computational/efficiency reasons, but also to make the transformer network be used to seeing contexts all the way from one to block_size.
# we'd like the transformer to be used to seeing everything in between. That will be useful later during inference because while we are sampling, we can start the sampling generation with as little as one character of context. 
# now, the transformer knows how to predict the next character with conext of just one. Then, it can predict everything up to block_size.
# after block_size, we have to start truncating because the transformer will never receive more than block size inputs when it is predicting the next character.

# we've looked at the time dimension of the tensors that are going to be feeding into the transformer. 
# we also care about the batch dimension. 
# As we are sampling the chunks of text and handing them to a transformer, we're going to have many batches of multiple chunks of text that are all "stacked up" in a single tensor (this is done for efficiency/to keep the GPUs busy, since they are great at parallel processing of data). 
# we want to process multiple chunks at the same time, but those chunks are processed completely independently. They don't talk to each other.

# generalize the code above and introduce a batch dimension:
torch.manual_seed(1337) # makes results reproducable
batch_size = 4 # how many independent sequences will we process in parallel? (every forwards and backwards pass of the transformer)
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # random offsets into the training set
    x = torch.stack([data[i:i+block_size] for i in ix]) # first block size characters
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # offset by 1 of x
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")

# there are 32 "examples" used for training in each batch
# the integer tensor (basically a matrix) of x (input values) is going to feed into the transformer. That transformer will simultaneously process all of the examples and then look up all of the correct integers to predict in tensor y
# NOW we have our batch of input that we'd like to feed into a transformer

print(xb) # our input to the transformer

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337) # makes results reproducable

# a bigram language model is the simplest possible nueral network
class BigramLanguageModel(nn.Module): 

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# https://www.youtube.com/watch?v=kCc8FmEb1nY
# 23:05
# https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=Q3k1Czf7LuA9


'''