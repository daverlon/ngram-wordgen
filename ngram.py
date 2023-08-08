import argparse
from itertools import islice
import numpy as np
from numpy.random import multinomial
from tqdm import tqdm

def prep_names(file, n=1):
  words = []
  start = "<" * (n-1) # padding
  end = ">"
  with open(file, 'r') as f:
    for line in f.readlines():
      words.append(start + line.replace("\n", "") + end)
  return words

if __name__ == "__main__":
  np.set_printoptions(suppress=True)
  #g = default_rng(1337)
  parser = argparse.ArgumentParser(description="N-gram model for name generation")
  parser.add_argument("-f", "--file", type=str, required=True, help="File to learn from")
  parser.add_argument("-n", "--ngram", type=int, required=True, help="Value of n for n-grams")
  parser.add_argument("-N", "--num_names", type=int, required=True, help="Value of n for n-grams")


  args = parser.parse_args()
  filename = args.file
  n = args.ngram
  k = args.num_names

  assert n >= 2

  words = prep_names(filename, n)
  chars = sorted(list(set(''.join(words))))
  #print(chars)
  l = len(chars)
  stoi = { s:i for i,s in enumerate(chars) }
  itos = { i:s for s,i in stoi.items() }

  ngrams = np.zeros((l,)*n)

  for word in tqdm(words, desc=f"Counting {n}-grams"):
    for group in zip(*(islice(word, i, None) for i in range(n))):
      c_values = group
      i_values = np.array([stoi[c] for c in c_values])
      pos = tuple(i_values[:n-1])
      val = tuple(i_values[n-1:])
      ngrams[pos][val] += 1

  its = np.prod(ngrams.shape[:-1])
  for idx in tqdm(np.ndindex(*ngrams.shape[:-1]), total=its, desc="Calculating probabilities"):
    s = ngrams[idx].sum()
    if s != 0.0:
      ngrams[idx] = ngrams[idx] / s

  stoi2 = lambda aa: tuple([stoi[a] for a in aa])

  outs = []

  for i in range(k):

    out = []
    for i in range(n-1): out.append('<')
    Cs = stoi2(out[-(n-1):])
    while True:
      Cn = multinomial(1, ngrams[Cs]).argmax()
      #print(Cn)
      out.append(itos[Cn])
      if Cn == 1: break # > (end)
      Cs = stoi2(out[-(n-1):])
    outs.append(''.join(out).replace('<','').replace('>',''))

  for word in outs:
    if word in [w.replace('<','').replace('>','') for w in words]:
      print(word, "✓")
    else:
      print(word)
