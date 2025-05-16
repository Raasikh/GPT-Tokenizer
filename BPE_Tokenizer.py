#!/usr/bin/env python
# coding: utf-8

# In[1]:


ord("R")


# In[6]:


list("hello raasikh)".encode("utf-8"))


# In[11]:


text="We do things a little differently! A senior team of managers and leaders from across the company form a virtual team of hiring leads to shepherd every application through our process. They are not recruiters, they run some aspect of the business, and we trust them to keep your interests and our business goals in mind and to find great opportunities in Canonical for the very best applicants around the world. You can email them a question, but we ask that you be gentle on their inboxes, and direct your questions to interviewers you meet along the way."
token= text.encode("utf-8")
token= list(map(int, token))
print(text)
print(len(text))
print(token)
print(len(token))


# In[19]:


chr(32), chr(116)


# In[20]:


def get_stats(ids):
    counts={}
    for pair in zip(ids, ids[1:]):
        counts[pair]= counts.get(pair, 0)+1
    return counts

stats=get_stats(token)
top_pair=max(stats, key=stats.get)
top_pair

# print(sorted(((v,k) for k,v in stats.items()), reverse=True))


# In[22]:


def merge(ids, pair, idx):
    newids=[]
    i=0
    while i< len(ids):
        if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            
            newids.append(idx)

            i+=2
        else:
                newids.append(ids[i])
                i+=1
    return newids

tokens2= merge(token,top_pair, 256)
print(tokens2)
print(len(tokens2))


# In[24]:


vocab_size = 276 # the desired final vocabulary size
num_merges = vocab_size - 256
ids = list(token) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx


# In[26]:


print("tokens length:", len(token))
print("ids length:", len(ids))
print(f"compression ratio: {len(token) / len(ids):.2f}X")


# In[ ]:


vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

print(decode([128]))


# In[ ]:


def encode(text):
  # given a string, return list of integers (the tokens)
  tokens = list(text.encode("utf-8"))
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens


# In[28]:


pip install tiktoken


# In[29]:


import tiktoken

# GPT-2 (does not merge spaces)
enc = tiktoken.get_encoding("gpt2")
print(enc.encode("    hello world!!!"))

# GPT-4 (merges spaces)
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode("    hello world!!!"))


# In[ ]:




