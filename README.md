# HackingGPT
## Part 6
Part 6 covers full self-attention with Query, Key, and Value projections, the complete attention mechanism with data-dependent weights, scaled attention to prevent peaky softmax, and layer normalization for training stability.

#### Author: [Kevin Thomas](mailto:ket189@pitt.edu)


```python
import torch
import torch.nn as nn
from torch.nn import functional as F
```


## Step 1: Load and Inspect the Data
Now let's read the file and see what we're working with. Understanding your data is crucial before building any model!


```python
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```


```python
text
```


**Output:**
```
'A dim glow rises behind the glass of a screen and the machine exhales in binary tides. The hum is a language and one who listens leans close to catch the quiet grammar. Patterns fold like small maps and seams hint at how the thing holds itself together. Treat each blinking diode and each idle tick as a sentence in a story that asks to be read.\n\nThere is patience here, not of haste but of careful unthreading. Where others see a sealed box the curious hand traces the join and wonders which thought made it fit. Do not rush to break, coax the meaning out with questions, and watch how the logic replies in traces and errors and in the echoes of forgotten interfaces.\n\nTechnology is artifact and argument at once. It makes a claim about what should be simple, what should be hidden, and what should be trusted. Reverse the gaze and learn its rhetoric, see where it promises ease, where it buries complexity, and where it leaves a backdoor as a sigh between bricks. To read that rhetoric is to be a kind interpreter, not a vandal.\n\nThis work is an apprenticeship in humility. Expect bafflement and expect to be corrected by small things, a timing oddity, a mismatch of expectation, a choice that favors speed over grace. Each misstep teaches a vocabulary of trade offs. Each discovery is a map of decisions and not a verdict on worth.\n\nThere is a moral keeping in the craft. Let curiosity be tempered with regard for consequence. Let repair and understanding lead rather than exploitation. The skill that opens a lock should also know when to hold the key and when to hand it back, mindful of harm and mindful of help.\n\nCelebrate the quiet victories, a stubborn protocol understood, an obscure format rendered speakable, a closed device coaxed into cooperation. These are small reconciliations between human intent and metal will, acts of translation rather than acts of conquest.\n\nAfter decoding a mechanism pause and ask what should change, a bug to be fixed, a user to be warned, a design to be amended. The true maker of machines leaves things better for having looked, not simply for having cracked the shell.'
```


## Step 2: Version 4 - FULL SELF-ATTENTION
Now we get to the real thing! In self-attention we have the following.

1. Each token produces a **Query** (Q): "What am I looking for?"
2. Each token produces a **Key** (K): "What do I contain?"
3. Each token produces a **Value** (V): "What information do I provide?"

The attention weights are computed as: **wei = Q @ K^T**
- High dot product equals tokens are "relevant" to each other.
- This is **data-dependent** where different inputs give different weights!

### Why Query, Key, Value?
Think of it like a search engine.
- **Query**: Your search terms (what you're looking for)
- **Key**: The titles/tags of documents (what each document contains)
- **Value**: The actual content of documents (what you get back)

The attention mechanism works in three steps.
1. Compute how well each query matches each key (dot product)
2. Normalize these scores to probabilities (softmax)
3. Use probabilities to weight the values (weighted sum)

### The Shape Journey
| Step | Tensor | Shape | Meaning |
|------|--------|-------|---------|
| Input | x | (B, T, C) | Batch of sequences with C features per token |
| Keys | k | (B, T, head_size) | Each token's "what I have" vector |
| Queries | q | (B, T, head_size) | Each token's "what I'm looking for" vector |
| Values | v | (B, T, head_size) | Each token's "what I'll give" vector |
| Raw Attention | wei | (B, T, T) | How much each position attends to each other position |
| Output | out | (B, T, head_size) | Weighted sum of values for each position |


```python
torch.manual_seed(42)
```


**Output:**
```
<torch._C.Generator at 0x117c8e4f0>
```


```python
# define batch dimension
B = 4  # batch size: 4 independent sequences
B
```


**Output:**
```
4
```


```python
# define time dimension
T = 8  # sequence length: 8 tokens/positions in each sequence
T
```


**Output:**
```
8
```


```python
# define channel dimension
C = 32  # feature size: 32 features per token
C
```


**Output:**
```
32
```


```python
# start with random data
x = torch.randn(B, T, C)
x
```


**Output:**
```
tensor([[[ 1.9269,  1.4873,  0.9007,  ...,  0.0418, -0.2516,  0.8599],
         [-1.3847, -0.8712, -0.2234,  ...,  1.8446, -1.1845,  1.3835],
         [ 1.4451,  0.8564,  2.2181,  ..., -0.8278,  1.3347,  0.4835],
         ...,
         [-1.9006,  0.2286,  0.0249,  ..., -0.5558,  0.7043,  0.7099],
         [ 1.7744, -0.9216,  0.9624,  ..., -0.5003,  1.0350,  1.6896],
         [-0.0045,  1.6668,  0.1539,  ...,  0.5655,  0.5058,  0.2225]],

        [[-0.6855,  0.5636, -1.5072,  ...,  1.1566,  0.2691, -0.0366],
         [ 0.9733, -1.0151, -0.5419,  ..., -0.0553,  1.2049, -0.9825],
         [ 0.4334, -0.7172,  1.0554,  ..., -0.6766, -0.5730, -0.3303],
         ...,
         [ 0.6839, -1.3246, -0.5161,  ...,  1.1895,  0.7607, -0.7463],
         [-1.3839,  0.4869, -1.0020,  ...,  1.9535,  2.0487, -1.0880],
         [ 1.6217,  0.8513, -0.4005,  ...,  0.4232, -0.3389,  0.5180]],

        [[-1.3638,  0.1930, -0.6103,  ...,  0.6110,  1.2208, -0.6076],
         [-1.7376, -0.1254, -1.3658,  ..., -0.6035, -0.1743,  0.6092],
         [-0.8032, -1.1209,  0.1956,  ...,  0.1598,  1.7698,  0.6268],
         ...,
         [ 2.1296, -1.5181,  0.1387,  ...,  3.0250,  1.3463,  0.8556],
         [ 0.3220,  0.4461,  1.5230,  ..., -1.4591, -1.4937, -0.2214],
         [ 0.2252, -0.0772,  0.9857,  ..., -1.6034, -0.4298,  0.5762]],

        [[ 0.3444, -3.1016, -1.4587,  ...,  1.4162,  0.6834, -0.1383],
         [ 0.9213,  0.5282, -0.0082,  ...,  2.1477, -0.6604,  0.1135],
         [-0.2206,  0.7118,  0.3416,  ...,  1.1383, -0.2505,  1.6705],
         ...,
         [ 0.0518, -0.3285, -2.2472,  ...,  1.4557, -0.3461, -0.2634],
         [-0.4477, -0.7288, -0.1607,  ...,  0.5405,  0.4351, -2.2717],
         [-0.1339, -0.0586,  0.1257,  ...,  1.1085,  0.5544,  1.5818]]])
```


```python
# head size: dimension of queries, keys, and values
head_size = 16
head_size
```


**Output:**
```
16
```


```python
# learnable linear transformation, projecting input to key vectors; "What I have."
key = nn.Linear(C, head_size, bias=False)
key
```


**Output:**
```
Linear(in_features=32, out_features=16, bias=False)
```


```python
# learnable linear transformation, projecting input to query vectors; "What I'm looking for?"
query = nn.Linear(C, head_size, bias=False)
query
```


**Output:**
```
Linear(in_features=32, out_features=16, bias=False)
```


```python
# learnable linear transformation, projecting input to value vectors; "What I'll give if queried."
value = nn.Linear(C, head_size, bias=False)
value
```


**Output:**
```
Linear(in_features=32, out_features=16, bias=False)
```


```python
# understand what nn.Linear does
print('understanding the linear projections')
print()
print('nn.Linear(C, head_size, bias=False) creates a matrix of shape (C, head_size)')
print(f'   C = {C} (input features)')
print(f'   head_size = {head_size} (output features)')
print()
print('when we call key(x), it computes: x @ key.weight.T')
print(f'   x shape: {x.shape} = (B={B}, T={T}, C={C})')
print(f'   key.weight shape: {key.weight.shape} = (head_size={head_size}, C={C})')
print(f'   key.weight.T shape: ({C}, {head_size})')
print()
print('matrix multiplication')
print(f'   (B, T, C) @ (C, head_size) = (B, T, head_size)')
print(f'   ({B}, {T}, {C}) @ ({C}, {head_size}) = ({B}, {T}, {head_size})')
print()
print('each token\'s C-dimensional vector gets projected to head_size dimensions')
print('these projections are LEARNED during training')
```


**Output:**
```
understanding the linear projections

nn.Linear(C, head_size, bias=False) creates a matrix of shape (C, head_size)
   C = 32 (input features)
   head_size = 16 (output features)

when we call key(x), it computes: x @ key.weight.T
   x shape: torch.Size([4, 8, 32]) = (B=4, T=8, C=32)
   key.weight shape: torch.Size([16, 32]) = (head_size=16, C=32)
   key.weight.T shape: (32, 16)

matrix multiplication
   (B, T, C) @ (C, head_size) = (B, T, head_size)
   (4, 8, 32) @ (32, 16) = (4, 8, 16)

each token's C-dimensional vector gets projected to head_size dimensions
these projections are LEARNED during training

```


### Step 1: Compute Keys


```python
# compute keys
k = key(x)  # (B, T, head_size) = (4, 8, 16)
k
```


**Output:**
```
tensor([[[ 7.0784e-02, -9.4861e-01, -5.9983e-01, -8.8679e-01,  4.7325e-02,
          -5.0741e-03, -6.7452e-03,  6.4850e-01, -3.2939e-01,  6.5462e-01,
          -3.5305e-01,  3.8077e-01,  3.3350e-01, -1.9763e-01, -1.5752e-01,
          -3.8165e-01],
         [-1.1677e+00,  2.7538e-01,  1.6652e+00, -2.7140e-01,  1.4043e-01,
           2.7449e-03, -1.0794e+00,  2.6188e-01, -1.1814e-01, -5.4476e-01,
           7.9574e-02, -3.6371e-02,  8.4531e-01,  8.1885e-01,  2.1071e-01,
          -5.8136e-01],
         [ 4.3110e-01, -6.2462e-01, -1.8344e-01, -2.9284e-01,  2.5957e-01,
          -5.1398e-01, -7.4316e-01,  1.2174e-01, -1.1386e+00,  4.8304e-01,
          -1.7443e-02,  6.1590e-01, -5.5573e-02, -1.0868e+00, -1.1277e+00,
           8.2533e-02],
         [-1.5513e-01,  3.9932e-01,  6.9395e-01, -1.9858e-01,  9.4391e-02,
          -1.5222e-01, -5.7460e-01,  5.1392e-01,  8.2661e-01,  1.7277e-02,
           3.8279e-01, -9.2914e-01,  3.2609e-01,  1.7348e-01,  1.0189e-01,
          -2.2036e-01],
         [ 1.8306e-01,  4.2543e-01,  4.3122e-01, -6.0842e-01,  6.0081e-01,
           3.6623e-01, -8.6503e-02, -9.6163e-02, -1.4067e-01, -2.3484e-01,
           5.9090e-01,  4.2985e-01, -6.2456e-01, -9.1971e-01,  1.2549e-01,
           6.8989e-01],
         [-3.7444e-01,  1.0035e+00,  9.7462e-01,  3.2149e-01, -6.8795e-01,
          -1.1626e+00, -5.8433e-01,  7.6088e-01,  4.3164e-01,  6.5277e-02,
           5.4637e-01, -1.6951e-01,  1.5960e-01,  1.8030e-02,  8.5434e-01,
          -8.5818e-01],
         [-9.3779e-01,  1.2550e+00,  1.2327e+00,  4.0494e-01, -8.2321e-01,
          -1.2013e-01, -8.0551e-01,  5.5824e-01,  2.4750e-01, -9.8268e-02,
           1.1595e+00,  9.4443e-02, -6.8768e-02, -5.7340e-02,  4.2941e-01,
           1.0696e-01],
         [ 4.4332e-01,  6.7758e-01, -7.4604e-01, -1.0522e-02, -2.5499e-02,
           1.2013e+00, -4.0023e-01,  5.1752e-01, -4.0429e-01,  5.1114e-01,
          -4.5502e-01, -6.9189e-01, -2.5769e-01,  7.8132e-01,  2.9977e-01,
          -2.0309e-01]],

        [[ 2.7949e-01,  3.8377e-01,  5.2063e-01, -5.3440e-01,  1.0310e+00,
           7.4677e-01, -5.8959e-01,  1.4523e+00,  4.3332e-01,  4.1400e-02,
           8.9054e-01, -2.5275e+00,  9.0357e-01,  6.1530e-01,  3.6645e-01,
          -7.7466e-01],
         [ 5.5106e-02,  6.2537e-01, -1.5956e-02,  2.7005e-01, -5.0123e-01,
          -9.2878e-01,  3.8933e-01, -1.2394e+00,  2.2684e-01,  4.0359e-01,
           5.6402e-02,  1.1318e+00, -1.4728e+00, -1.4327e-01, -1.8886e-01,
           6.7431e-01],
         [-3.9762e-01, -1.6270e-01, -1.8589e-01,  4.7662e-02, -2.0473e-01,
           4.1716e-01, -1.8120e-01,  8.6617e-01, -5.7387e-01, -1.3463e-01,
           1.1304e-01,  2.0634e-01,  2.0002e-01, -1.4251e-01,  3.2072e-01,
           4.8422e-01],
         [-7.1437e-01,  2.3439e-01,  5.0224e-01,  3.1803e-01,  3.9729e-01,
           8.6735e-02, -1.1195e-01,  4.7673e-01, -1.7878e-01,  2.8000e-01,
          -7.3027e-01,  4.8926e-01, -6.1345e-01, -3.5076e-01, -4.3743e-01,
           6.1672e-01],
         [ 8.8705e-01, -7.7208e-01, -9.7915e-01,  1.5324e-02,  1.0060e-01,
           4.7434e-02, -8.7970e-02, -7.6677e-01, -6.2420e-01,  4.1531e-01,
          -1.5666e-01,  5.4941e-01, -6.7704e-02, -3.3108e-01, -5.1641e-01,
          -8.0151e-02],
         [ 1.0009e+00,  1.6927e-01,  4.2891e-02, -3.0627e-01, -3.1331e-01,
          -1.1513e-02,  3.7705e-01, -4.0904e-01,  2.8013e-01,  4.4826e-01,
          -1.8397e-01,  1.0289e-01, -3.9065e-01, -1.8056e-01, -5.4006e-01,
           1.0496e-01],
         [-7.9961e-01,  5.5213e-01,  4.5932e-01,  2.5351e-01, -4.8840e-01,
           2.8189e-01, -1.8903e-01,  4.2532e-01,  4.9572e-02, -4.2261e-01,
          -7.9938e-01,  1.5264e-01, -3.0169e-01,  7.4319e-01,  1.0023e+00,
          -6.4789e-01],
         [ 7.2109e-02, -2.5520e-01, -1.3666e+00, -8.5867e-01, -5.4527e-01,
          -1.5793e-01,  6.0183e-01,  4.0682e-01, -5.7640e-01,  1.6719e-01,
          -7.1796e-01,  4.8087e-01, -2.2927e-01,  6.8334e-01,  4.9681e-01,
           1.2973e-01]],

        [[-3.0747e-01, -8.4684e-01, -2.0854e-01, -7.0981e-01,  2.2592e-01,
           8.0769e-01, -8.7980e-01,  5.2975e-01, -5.3864e-01, -6.9849e-01,
           3.5470e-01,  3.9033e-01, -8.8904e-02, -1.9965e-01,  5.3527e-01,
          -1.2782e-01],
         [ 2.4880e-02, -6.1044e-01,  5.6097e-01,  9.3349e-01,  6.7494e-01,
           5.3337e-01,  3.8625e-01,  4.1990e-01, -6.3053e-01, -1.6064e-01,
           7.5473e-01, -2.0709e-01, -1.2627e-01, -1.0944e-01,  1.7607e-01,
           2.9459e-01],
         [ 4.0286e-01,  5.0442e-01,  5.5335e-01,  1.3052e-01, -2.4466e-01,
           3.9971e-01, -1.5004e-01,  6.9629e-01, -2.6634e-01, -1.9435e-01,
           6.9368e-02, -6.8080e-01, -8.7913e-02,  1.4587e+00, -5.7588e-01,
          -7.8258e-02],
         [-2.1512e-01,  3.2160e-01,  8.3376e-01,  8.4302e-02, -1.5945e-01,
          -1.1895e+00, -4.4709e-01,  5.2719e-01,  5.9774e-01, -3.7033e-01,
           9.9006e-02, -5.7947e-01,  3.1496e-01,  1.0079e+00,  1.0719e+00,
          -5.6954e-01],
         [-2.8004e-01, -2.0767e-01,  2.5970e-01,  8.4324e-01,  2.5237e-01,
          -8.8464e-01, -1.0020e-01, -4.5292e-01, -8.3785e-01, -6.2902e-03,
          -8.9628e-01,  5.6377e-01,  4.7216e-01, -1.6173e-01, -3.5555e-01,
          -1.1371e-01],
         [ 1.0536e+00,  1.3793e+00,  6.2686e-01, -1.4053e+00, -5.8837e-01,
           9.5645e-01,  1.6153e-01,  4.8475e-01,  8.2436e-01,  9.8684e-01,
          -5.6398e-01, -7.7153e-01, -6.8535e-01,  8.9388e-01, -6.0170e-01,
           6.5138e-01],
         [-9.0461e-01, -3.9238e-01,  2.7969e-01, -1.1222e+00, -1.6980e-01,
          -1.2657e-01,  2.4860e-01, -3.6713e-02, -6.6537e-02,  4.1127e-01,
           1.4832e-01,  8.3464e-02,  3.6814e-01, -1.0768e+00,  3.0961e-02,
           3.7589e-01],
         [ 7.0817e-01, -1.2292e+00, -1.1683e+00, -4.6781e-01,  4.1275e-01,
           1.7956e-01,  3.1878e-01, -6.3456e-01,  2.6772e-01,  8.7278e-01,
           2.8422e-01,  1.0724e-01,  4.0734e-01, -1.4641e+00, -7.2267e-01,
           6.8318e-01]],

        [[ 2.6565e-01,  5.8151e-01,  5.1564e-02, -3.7245e-03,  1.8873e-02,
           7.3230e-01,  2.6759e-01,  1.4722e+00, -4.7369e-02, -9.7170e-01,
           3.1321e-01, -6.1068e-01,  8.4183e-01,  1.9014e-01, -6.2080e-01,
           5.6473e-01],
         [-3.7211e-01, -4.3067e-01,  5.9307e-01, -1.0694e+00,  1.3655e-01,
           1.0658e-01, -5.9351e-01,  1.8878e-01,  1.6176e-01, -1.4650e-01,
           4.3989e-01,  3.7046e-01,  1.3988e-01, -3.9782e-01, -1.2492e-01,
          -4.1166e-01],
         [-4.6137e-01,  2.0230e-02,  2.1769e-01, -6.4556e-01, -1.9675e-01,
          -5.6151e-01,  2.5675e-01, -1.8731e-01,  1.8114e-01, -6.7379e-01,
          -1.0939e+00,  6.0911e-01,  3.5566e-01,  5.7098e-01, -7.0692e-01,
           1.0351e-01],
         [ 6.4570e-02, -9.8439e-01,  2.1197e-02,  1.3539e-01,  6.8128e-01,
          -2.7760e-01,  4.6577e-01,  3.8781e-02,  9.2859e-02, -9.4311e-01,
           7.1776e-01,  9.6358e-02,  9.0489e-01, -3.7272e-01,  4.5790e-01,
          -4.6718e-01],
         [ 2.8714e-01,  1.0547e+00,  3.7973e-02, -5.5168e-01, -5.6647e-01,
           2.0369e-02,  4.5265e-01,  1.0909e-01,  2.9652e-01, -6.2592e-02,
          -4.7420e-02, -6.3049e-01, -2.2637e-01,  1.8829e+00, -3.6382e-01,
           1.0184e+00],
         [-4.4120e-02,  5.2540e-01, -4.7653e-01,  8.1647e-01, -9.2644e-02,
           3.2372e-01, -5.9504e-03, -3.0648e-03, -2.4775e-01, -9.3111e-01,
          -3.5523e-01,  2.7222e-01,  1.2949e-01,  2.3470e-01,  3.6988e-01,
           1.0492e-01],
         [-2.7105e-01,  2.0783e-01,  6.0952e-01, -8.4854e-01,  9.0097e-02,
          -7.3273e-01, -3.5330e-01, -7.2682e-01,  6.9166e-02, -4.4932e-01,
          -4.6002e-04,  3.1294e-01, -1.4925e-01,  1.1107e+00,  3.0462e-01,
          -1.9615e-01],
         [-3.0565e-01,  5.9027e-01, -8.9811e-02, -2.6312e-01, -5.1747e-02,
          -3.0924e-01,  2.4340e-02,  8.7760e-01,  5.4526e-01, -3.1863e-01,
          -1.7977e-01, -9.8312e-01,  8.7145e-01,  5.3252e-01, -1.1281e+00,
           8.0423e-02]]], grad_fn=<UnsafeViewBackward0>)
```


```python
# k shape
k.shape
```


**Output:**
```
torch.Size([4, 8, 16])
```


```python
# understand the keys tensor
print('understanding the keys (k)')
print()
print(f'k shape: {k.shape} = (B={B}, T={T}, head_size={head_size})')
print()
print('what does each dimension mean?')
print(f'   B={B}: we have {B} independent sequences (batch)')
print(f'   T={T}: each sequence has {T} tokens/positions')
print(f'   head_size={head_size}: each key vector has {head_size} dimensions')
print()
print('for batch 0, position 0')
print(f'   k[0, 0] = {k[0, 0].tolist()[:4]}... (first 4 of {head_size} values)')
print('   this is position 0\'s "what I contain" vector')
print()
print('for batch 0, position 1')
print(f'   k[0, 1] = {k[0, 1].tolist()[:4]}... (first 4 of {head_size} values)')
print('   this is position 1\'s "what I contain" vector')
print()
print('each position has its own unique key vector')
print('keys tell us what information each token has to offer')
```


**Output:**
```
understanding the keys (k)

k shape: torch.Size([4, 8, 16]) = (B=4, T=8, head_size=16)

what does each dimension mean?
   B=4: we have 4 independent sequences (batch)
   T=8: each sequence has 8 tokens/positions
   head_size=16: each key vector has 16 dimensions

for batch 0, position 0
   k[0, 0] = [0.07078436762094498, -0.9486113786697388, -0.5998315811157227, -0.8867908120155334]... (first 4 of 16 values)
   this is position 0's "what I contain" vector

for batch 0, position 1
   k[0, 1] = [-1.1676945686340332, 0.27537861466407776, 1.6651532649993896, -0.271398663520813]... (first 4 of 16 values)
   this is position 1's "what I contain" vector

each position has its own unique key vector
keys tell us what information each token has to offer

```


### Step 2: Compute Queries


```python
# computer queries
q = query(x)  # (B, T, head_size) = (4, 8, 16)
q
```


**Output:**
```
tensor([[[-0.4467,  0.6160, -0.5806, -0.4032, -1.2520, -0.3569,  0.3134,
          -0.7361,  0.9993,  0.0667, -0.5147,  0.0405,  0.2954, -0.2314,
          -0.5547,  0.0771],
         [-0.5397,  0.7083, -0.7480, -0.7873,  0.8294, -0.5856,  0.1124,
          -0.3189,  0.3401, -0.4997,  0.9458, -0.9430,  0.4883,  0.3419,
          -0.2482, -0.1744],
         [-0.6314,  0.3053, -0.8942, -0.2582, -1.1889, -0.5720, -0.1693,
          -0.6428,  0.3851, -0.3830, -0.7606,  0.9779, -0.4845, -0.7168,
          -0.1794, -0.3390],
         [-0.4691, -0.1064,  0.0868, -0.1737,  0.3330,  0.1009,  0.0584,
          -0.0182,  0.1981, -0.0587,  0.4190, -0.6249,  0.1330,  0.0505,
          -0.7110, -0.1839],
         [ 0.9760,  0.0929, -0.5380, -0.2953, -0.6723,  0.3561,  0.2937,
          -0.0166, -0.4360, -0.4318, -1.0178, -1.0906, -0.0714,  0.4877,
           0.9537,  0.1052],
         [-0.2460,  0.1963, -0.0330, -0.0296,  0.4262,  0.5681, -0.5607,
           0.3858,  0.6359, -0.1462,  0.9695,  0.5166,  0.0834,  0.6468,
          -0.6654,  0.6578],
         [ 0.1025, -0.6487, -1.0766,  0.5657, -0.5930,  0.7750, -0.3525,
           0.5930, -0.7849, -0.0991,  0.1253,  0.2647,  0.0772,  0.1080,
          -0.3049,  0.7578],
         [ 0.0623,  0.0865,  0.4167,  0.4756, -0.6816, -0.5767, -0.0152,
          -0.0497,  0.4178,  0.6291,  0.0292, -0.3211, -0.9214,  0.4931,
           0.4157,  0.3642]],

        [[ 0.0282, -0.4232, -0.0589,  0.2576, -0.5406,  0.1493, -0.0072,
          -0.1857,  0.2729,  0.0872,  0.5680, -0.3410, -0.8786,  0.1489,
          -0.3439, -0.5695],
         [ 0.3350, -0.0913,  0.3372,  0.5263,  0.2454,  0.7274, -1.1037,
           1.2901, -0.2334,  0.2306, -0.2329,  1.0960, -0.5200, -0.2347,
           0.5013,  0.1544],
         [ 0.3323, -0.3075, -0.5424, -0.2402, -0.6157, -0.2720,  0.2610,
          -0.1437, -0.6736, -0.4015, -0.4294, -0.2657,  1.3191, -0.0347,
           0.3950, -0.0393],
         [ 0.5122,  0.4843,  0.4991, -0.2277,  0.8109,  0.4206, -0.2067,
           0.6427,  0.3992, -0.5464, -0.3880, -0.2444,  0.6319,  0.8060,
           0.3155, -0.3900],
         [-0.6884,  0.1610, -0.9344, -0.0771, -0.7246, -1.3584,  0.2215,
          -0.4750,  0.5761, -0.1267,  0.3564, -0.0592, -0.7735, -0.3088,
           0.1279,  0.3532],
         [-0.6722,  0.3934, -0.1227, -0.3349,  0.1176,  0.2358,  0.4627,
          -0.6571, -0.1819,  0.3838,  0.1175,  0.9820, -0.5690, -0.0073,
          -0.2621,  0.4604],
         [ 0.2283, -0.4608,  0.5600,  0.1657,  1.2347,  0.0294,  0.3925,
          -0.0448,  0.1652, -0.7090,  0.3402,  0.5073,  0.4177,  0.8930,
          -0.1127, -0.7572],
         [-0.3415,  0.4639, -0.0265, -0.0486, -0.1049, -1.0631, -0.2458,
          -0.1110,  0.6402,  0.4874, -0.0171,  0.4958, -0.5030,  0.6007,
           0.4187,  0.6639]],

        [[ 0.9324, -0.3873,  0.7694,  0.2836,  0.0797, -0.5609, -0.0542,
           0.0164, -0.3471, -0.6118, -0.6670,  0.0451,  1.4065,  0.4547,
           0.7767, -0.5672],
         [ 1.4794,  0.4999,  0.5846, -0.0675,  0.1716, -0.3754,  1.4422,
           0.3717, -0.0733, -0.2583,  0.1913, -0.0275,  0.3572,  0.7829,
           0.0142, -1.2853],
         [ 0.6440, -0.0962,  0.4457,  0.8338, -0.3444, -0.3782, -0.0274,
          -0.0371,  0.5641,  1.0235,  0.1433,  0.6200, -0.5572,  0.5647,
           0.0929, -0.6358],
         [-0.5921,  0.2062, -0.0420,  0.2374,  0.7923,  0.6848, -0.2879,
           0.6016,  1.3414, -0.4967,  1.6375, -0.5447, -1.2369,  0.8092,
          -1.1227, -0.2975],
         [-0.8941, -0.2259, -0.0089, -0.2834,  0.5162, -0.1793, -0.7540,
           0.0698,  0.4712, -0.6827, -0.3188, -0.1305,  0.2372, -0.4696,
           0.1738, -0.6162],
         [ 0.2228,  0.2739, -0.3661, -0.3544, -0.1279,  0.1681,  0.3655,
          -1.0201,  0.0591,  1.3898,  0.0566,  0.3300, -1.5571,  0.4145,
          -0.6352,  0.3029],
         [-0.2113,  0.1152, -0.7422, -1.1339,  0.4245,  0.0725,  0.6782,
          -0.6515, -0.6226, -0.6749, -0.3655, -0.0432,  0.7225, -0.4106,
          -0.5273, -0.3364],
         [-0.5688,  0.4958, -0.2908, -0.2989, -0.4141, -2.0439,  0.9589,
          -0.4933, -0.5134, -0.6563, -0.4388, -0.0541,  0.3163, -0.6713,
          -0.1591, -0.2476]],

        [[ 0.0763, -0.7024,  0.1258, -0.1102,  0.2028, -0.0728,  0.9231,
          -0.1842, -0.3658,  0.2163, -0.2010, -0.0635,  1.2323,  0.6367,
           0.7457,  0.8029],
         [-0.1333, -0.1337, -0.3907,  0.3097, -0.4754,  0.6433, -1.5990,
           0.2370,  0.7829, -0.2601, -0.9853,  0.1865, -0.0083,  0.3066,
           0.1418,  0.3310],
         [-0.7890,  0.5769, -0.2716, -0.3727,  0.5939, -0.3055,  0.4481,
          -1.1364,  0.7368,  0.1956,  0.3569,  0.0049, -0.1753,  0.1731,
          -0.4684,  0.0436],
         [ 0.0596, -0.4374, -0.1358,  0.2870, -0.8057, -0.1064, -0.1653,
           0.4008,  0.5390, -0.1343,  1.0012,  0.3865,  0.9892,  0.4055,
           0.4856,  0.7514],
         [ 0.5913, -0.3046,  0.5919,  1.5844, -0.7266,  0.6204, -0.7069,
           1.2992,  0.5931,  1.1558,  0.4605, -0.1100, -0.4958,  0.1390,
           0.4130, -0.2124],
         [-0.3426, -0.5883,  0.2461,  0.4987,  0.2900,  0.2267,  0.2203,
          -0.0994, -0.1966, -0.0249,  0.4232, -0.4375,  0.1825,  0.1140,
           0.4494,  0.7788],
         [ 0.4462,  0.0769,  0.3909,  0.5763, -0.6788,  1.3467, -1.1254,
           1.5560,  0.3330, -0.2107,  0.1921, -0.4350,  0.8739,  0.1294,
           1.5133,  0.0672],
         [-0.6450, -0.1364,  0.1006,  0.1115,  0.1079,  0.7784, -0.0366,
          -0.4395,  0.5661,  0.8980,  0.2271,  0.0808, -0.0898, -0.5526,
          -0.8941,  0.6091]]], grad_fn=<UnsafeViewBackward0>)
```


```python
# query shape
q.shape
```


**Output:**
```
torch.Size([4, 8, 16])
```


```python
# understand the queries tensor
print('understanding the queries (q)')
print()
print(f'q shape: {q.shape} = (B={B}, T={T}, head_size={head_size})')
print()
print('what does each dimension mean?')
print(f'   B={B}: we have {B} independent sequences (batch)')
print(f'   T={T}: each sequence has {T} tokens/positions')
print(f'   head_size={head_size}: each query vector has {head_size} dimensions')
print()
print('for batch 0, position 0')
print(f'   q[0, 0] = {q[0, 0].tolist()[:4]}... (first 4 of {head_size} values)')
print('   this is position 0\'s "what I\'m looking for" vector')
print()
print('for batch 0, position 1')
print(f'   q[0, 1] = {q[0, 1].tolist()[:4]}... (first 4 of {head_size} values)')
print('   this is position 1\'s "what I\'m looking for" vector')
print()
print('each position has its own unique query vector')
print('queries tell us what information each token is searching for')
```


**Output:**
```
understanding the queries (q)

q shape: torch.Size([4, 8, 16]) = (B=4, T=8, head_size=16)

what does each dimension mean?
   B=4: we have 4 independent sequences (batch)
   T=8: each sequence has 8 tokens/positions
   head_size=16: each query vector has 16 dimensions

for batch 0, position 0
   q[0, 0] = [-0.44672077894210815, 0.6160160899162292, -0.580593466758728, -0.4032447636127472]... (first 4 of 16 values)
   this is position 0's "what I'm looking for" vector

for batch 0, position 1
   q[0, 1] = [-0.5397410988807678, 0.7082756161689758, -0.7479986548423767, -0.7873462438583374]... (first 4 of 16 values)
   this is position 1's "what I'm looking for" vector

each position has its own unique query vector
queries tell us what information each token is searching for

```


### Step 3: Compute Raw Attention Scores


```python
# compute raw attention scores (affinities)
# wei[b, i, j] = how much position i attends to position j
# computed as dot product of query[i] and key[j]
wei = q @ k.transpose(-2, -1)  # (B, T, 16) @ (B, 16, T) → (B, T, T)
wei
```


**Output:**
```
tensor([[[-0.3332, -1.1723, -1.0216, -0.0545, -1.0950,  0.2735,  0.1340,
          -0.8490],
         [-0.6597,  0.7869, -1.2725,  1.6851,  0.1159,  0.5450,  0.2356,
          -0.1962],
         [ 0.3630, -1.5219,  0.7821, -1.7215, -0.3494,  0.2884, -0.1021,
          -1.4271],
         [-0.1001,  0.8649, -0.0335,  1.0221, -0.1350, -0.3078,  0.1440,
          -0.3019],
         [ 0.0136, -1.6202, -1.9888, -0.3327, -1.2506, -0.8928, -2.2674,
           3.0561],
         [-0.5833,  1.2025, -0.3281,  0.9147,  0.9809, -0.4859,  1.7589,
           0.1650],
         [ 1.1351, -1.9940,  1.5545, -1.8037, -0.5062, -2.6109, -1.0739,
           1.6430],
         [-1.2784, -0.4554, -1.4118,  0.6392, -0.5780,  1.9291,  1.6689,
           0.1103]],

        [[ 0.0685,  0.8637, -0.6632, -0.6766,  0.3320,  0.5343, -0.1189,
          -0.4954],
         [-0.3457, -0.6529,  1.8824,  2.0650, -0.0657, -0.6715,  1.4419,
          -0.1583],
         [-0.0532, -2.0680,  0.5618, -1.8266,  0.9683, -0.3595, -0.1953,
           1.8939],
         [ 4.8230, -3.2096, -0.1085, -0.4954, -1.7757, -0.6045,  1.5651,
          -0.5931],
         [-3.5622,  3.7869, -0.6365, -0.3043, -0.0541,  0.2013, -0.3909,
           1.2404],
         [-4.2631,  3.2677, -0.0738,  1.3912,  0.6345,  0.5088, -0.2544,
           0.8281],
         [ 1.6236, -1.5654, -0.7652, -0.7248, -0.2393, -0.7221,  0.2028,
          -1.0873],
         [-2.2400,  3.2660, -0.4738,  0.7659, -0.9465,  0.1319,  0.7583,
           0.7500]],

        [[-0.0314,  0.2432,  0.6900,  3.0115,  2.0694, -1.9951, -1.2359,
          -1.8439],
         [-2.0277,  0.3562,  2.3180,  2.0301, -0.3015,  1.8700, -2.2014,
          -2.0738],
         [-1.8383, -0.0600,  0.5855,  1.2383,  0.3526,  0.8046, -1.8479,
          -0.9355],
         [ 0.5299,  1.4638,  2.5979,  0.5391, -3.5146,  1.8997, -1.7077,
          -1.5062],
         [ 1.6449, -0.6936, -1.2360,  1.3106,  0.4067, -2.5170,  0.9544,
          -0.6532],
         [-1.9045, -1.1392, -0.1105, -2.7667, -0.8025,  3.8325, -0.2067,
           1.8609],
         [ 0.5785, -1.2410, -1.2474, -2.1258,  0.6188, -1.0260,  1.5934,
           2.1066],
         [-2.2108, -1.9236, -2.0073,  1.1402,  2.6994, -3.1374,  1.5955,
          -0.2352]],

        [[ 0.4770, -0.8036,  0.6485,  1.7210,  1.0219,  0.0428,  0.3335,
          -0.0809],
         [ 0.1920,  0.0650,  0.6659, -1.6642,  0.2617,  1.2427, -0.0642,
           0.4364],
         [-1.5678,  0.0473,  0.9261, -0.3383,  1.0800, -0.6310,  1.4611,
           0.5233],
         [ 1.4212,  0.1287, -0.5043,  1.4374,  0.8206,  0.3293, -0.1394,
           0.4498],
         [ 0.4617, -0.8008, -3.4562, -1.3081, -0.5987, -0.2240, -2.5389,
          -0.7223],
         [ 0.4312, -0.4930, -1.0866,  1.0946, -0.0532,  0.1422, -0.5041,
          -0.5325],
         [ 3.5177,  0.2910, -2.6452,  0.4690, -0.0807,  1.4431, -1.6943,
           0.4496],
         [-0.4898,  0.4337, -0.4948, -1.2595, -0.5212, -1.1997, -1.4466,
           0.0328]]], grad_fn=<UnsafeViewBackward0>)
```


```python
# understand the attention score computation
print('understanding q @ k.transpose(-2, -1)')
print()
print(f'q shape: {q.shape} = (B, T, head_size) = ({B}, {T}, {head_size})')
print(f'k shape: {k.shape} = (B, T, head_size) = ({B}, {T}, {head_size})')
print()
print('k.transpose(-2, -1) swaps the last two dimensions')
print(f'   before: k shape = ({B}, {T}, {head_size})')
print(f'   after:  k.T shape = ({B}, {head_size}, {T})')
print()
print('the matrix multiplication')
print(f'   (B, T, head_size) @ (B, head_size, T) = (B, T, T)')
print(f'   ({B}, {T}, {head_size}) @ ({B}, {head_size}, {T}) = ({B}, {T}, {T})')
print()
print('what does wei[b, i, j] mean?')
print('   wei[b, i, j] = dot product of query[i] and key[j] in batch b')
print('   higher value = position i is more interested in position j')
print('   lower value = position i is less interested in position j')
print()
print(f'wei shape: {wei.shape}')
```


**Output:**
```
understanding q @ k.transpose(-2, -1)

q shape: torch.Size([4, 8, 16]) = (B, T, head_size) = (4, 8, 16)
k shape: torch.Size([4, 8, 16]) = (B, T, head_size) = (4, 8, 16)

k.transpose(-2, -1) swaps the last two dimensions
   before: k shape = (4, 8, 16)
   after:  k.T shape = (4, 16, 8)

the matrix multiplication
   (B, T, head_size) @ (B, head_size, T) = (B, T, T)
   (4, 8, 16) @ (4, 16, 8) = (4, 8, 8)

what does wei[b, i, j] mean?
   wei[b, i, j] = dot product of query[i] and key[j] in batch b
   higher value = position i is more interested in position j
   lower value = position i is less interested in position j

wei shape: torch.Size([4, 8, 8])

```


```python
# trace through one attention score calculation manually
print('tracing through one attention score calculation')
print()
print('let\'s compute wei[0, 0, 1] manually')
print('this is: how much does position 0 attend to position 1 (in batch 0)?')
print()
print(f'q[0, 0] (query at position 0):')
print(f'   {q[0, 0].tolist()[:4]}... (first 4 values)')
print()
print(f'k[0, 1] (key at position 1):')
print(f'   {k[0, 1].tolist()[:4]}... (first 4 values)')
print()
print('dot product = sum of element-wise products')
manual_dot = (q[0, 0] * k[0, 1]).sum().item()
print(f'   q[0, 0] · k[0, 1] = {manual_dot:.4f}')
print()
print(f'actual wei[0, 0, 1] = {wei[0, 0, 1].item():.4f}')
print(f'match: {abs(manual_dot - wei[0, 0, 1].item()) < 1e-5}')
```


**Output:**
```
tracing through one attention score calculation

let's compute wei[0, 0, 1] manually
this is: how much does position 0 attend to position 1 (in batch 0)?

q[0, 0] (query at position 0):
   [-0.44672077894210815, 0.6160160899162292, -0.580593466758728, -0.4032447636127472]... (first 4 values)

k[0, 1] (key at position 1):
   [-1.1676945686340332, 0.27537861466407776, 1.6651532649993896, -0.271398663520813]... (first 4 values)

dot product = sum of element-wise products
   q[0, 0] · k[0, 1] = -1.1723

actual wei[0, 0, 1] = -1.1723
match: True

```


```python
# examine one row of attention scores (before masking)
print('examining raw attention scores for position 0 (batch 0)')
print()
print(f'wei[0, 0] = attention scores from position 0 to all positions')
print(f'   {wei[0, 0].tolist()}')
print()
print('interpreting these scores')
for j in range(T):
    score = wei[0, 0, j].item()
    if score > 0:
        print(f'   position 0 → position {j}: {score:.4f} (positive = interested)')
    else:
        print(f'   position 0 → position {j}: {score:.4f} (negative = not interested)')
print()
print('note: these are RAW scores, not probabilities yet')
print('we still need to apply masking and softmax')
```


**Output:**
```
examining raw attention scores for position 0 (batch 0)

wei[0, 0] = attention scores from position 0 to all positions
   [-0.33323800563812256, -1.1722666025161743, -1.0215535163879395, -0.05452167987823486, -1.0950181484222412, 0.27353763580322266, 0.13403844833374023, -0.8489717841148376]

interpreting these scores
   position 0 → position 0: -0.3332 (negative = not interested)
   position 0 → position 1: -1.1723 (negative = not interested)
   position 0 → position 2: -1.0216 (negative = not interested)
   position 0 → position 3: -0.0545 (negative = not interested)
   position 0 → position 4: -1.0950 (negative = not interested)
   position 0 → position 5: 0.2735 (positive = interested)
   position 0 → position 6: 0.1340 (positive = interested)
   position 0 → position 7: -0.8490 (negative = not interested)

note: these are RAW scores, not probabilities yet
we still need to apply masking and softmax

```


### Step 4: Mask Future Positions


```python
# create lower-triangular mask
tril = torch.tril(torch.ones(T, T))
tril
```


**Output:**
```
tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1.]])
```


```python
# understand the lower triangular mask
print('understanding the lower triangular mask')
print()
print('tril = torch.tril(torch.ones(T, T))')
print(f'T = {T}, so we create an {T}x{T} matrix')
print()
print('examining each row of tril')
for i in range(T):
    row = tril[i].tolist()
    visible = [j for j, v in enumerate(row) if v == 1.0]
    print(f'row {i}: {[int(v) for v in row]}')
    print(f'       position {i} can see positions {visible}')
    print()
```


**Output:**
```
understanding the lower triangular mask

tril = torch.tril(torch.ones(T, T))
T = 8, so we create an 8x8 matrix

examining each row of tril
row 0: [1, 0, 0, 0, 0, 0, 0, 0]
       position 0 can see positions [0]

row 1: [1, 1, 0, 0, 0, 0, 0, 0]
       position 1 can see positions [0, 1]

row 2: [1, 1, 1, 0, 0, 0, 0, 0]
       position 2 can see positions [0, 1, 2]

row 3: [1, 1, 1, 1, 0, 0, 0, 0]
       position 3 can see positions [0, 1, 2, 3]

row 4: [1, 1, 1, 1, 1, 0, 0, 0]
       position 4 can see positions [0, 1, 2, 3, 4]

row 5: [1, 1, 1, 1, 1, 1, 0, 0]
       position 5 can see positions [0, 1, 2, 3, 4, 5]

row 6: [1, 1, 1, 1, 1, 1, 1, 0]
       position 6 can see positions [0, 1, 2, 3, 4, 5, 6]

row 7: [1, 1, 1, 1, 1, 1, 1, 1]
       position 7 can see positions [0, 1, 2, 3, 4, 5, 6, 7]


```


```python
# apply the mask to the attention scores
wei = wei.masked_fill(tril == 0, float('-inf'))
wei
```


**Output:**
```
tensor([[[-0.3332,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,
             -inf],
         [-0.6597,  0.7869,    -inf,    -inf,    -inf,    -inf,    -inf,
             -inf],
         [ 0.3630, -1.5219,  0.7821,    -inf,    -inf,    -inf,    -inf,
             -inf],
         [-0.1001,  0.8649, -0.0335,  1.0221,    -inf,    -inf,    -inf,
             -inf],
         [ 0.0136, -1.6202, -1.9888, -0.3327, -1.2506,    -inf,    -inf,
             -inf],
         [-0.5833,  1.2025, -0.3281,  0.9147,  0.9809, -0.4859,    -inf,
             -inf],
         [ 1.1351, -1.9940,  1.5545, -1.8037, -0.5062, -2.6109, -1.0739,
             -inf],
         [-1.2784, -0.4554, -1.4118,  0.6392, -0.5780,  1.9291,  1.6689,
           0.1103]],

        [[ 0.0685,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,
             -inf],
         [-0.3457, -0.6529,    -inf,    -inf,    -inf,    -inf,    -inf,
             -inf],
         [-0.0532, -2.0680,  0.5618,    -inf,    -inf,    -inf,    -inf,
             -inf],
         [ 4.8230, -3.2096, -0.1085, -0.4954,    -inf,    -inf,    -inf,
             -inf],
         [-3.5622,  3.7869, -0.6365, -0.3043, -0.0541,    -inf,    -inf,
             -inf],
         [-4.2631,  3.2677, -0.0738,  1.3912,  0.6345,  0.5088,    -inf,
             -inf],
         [ 1.6236, -1.5654, -0.7652, -0.7248, -0.2393, -0.7221,  0.2028,
             -inf],
         [-2.2400,  3.2660, -0.4738,  0.7659, -0.9465,  0.1319,  0.7583,
           0.7500]],

        [[-0.0314,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,
             -inf],
         [-2.0277,  0.3562,    -inf,    -inf,    -inf,    -inf,    -inf,
             -inf],
         [-1.8383, -0.0600,  0.5855,    -inf,    -inf,    -inf,    -inf,
             -inf],
         [ 0.5299,  1.4638,  2.5979,  0.5391,    -inf,    -inf,    -inf,
             -inf],
         [ 1.6449, -0.6936, -1.2360,  1.3106,  0.4067,    -inf,    -inf,
             -inf],
         [-1.9045, -1.1392, -0.1105, -2.7667, -0.8025,  3.8325,    -inf,
             -inf],
         [ 0.5785, -1.2410, -1.2474, -2.1258,  0.6188, -1.0260,  1.5934,
             -inf],
         [-2.2108, -1.9236, -2.0073,  1.1402,  2.6994, -3.1374,  1.5955,
          -0.2352]],

        [[ 0.4770,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,
             -inf],
         [ 0.1920,  0.0650,    -inf,    -inf,    -inf,    -inf,    -inf,
             -inf],
         [-1.5678,  0.0473,  0.9261,    -inf,    -inf,    -inf,    -inf,
             -inf],
         [ 1.4212,  0.1287, -0.5043,  1.4374,    -inf,    -inf,    -inf,
             -inf],
         [ 0.4617, -0.8008, -3.4562, -1.3081, -0.5987,    -inf,    -inf,
             -inf],
         [ 0.4312, -0.4930, -1.0866,  1.0946, -0.0532,  0.1422,    -inf,
             -inf],
         [ 3.5177,  0.2910, -2.6452,  0.4690, -0.0807,  1.4431, -1.6943,
             -inf],
         [-0.4898,  0.4337, -0.4948, -1.2595, -0.5212, -1.1997, -1.4466,
           0.0328]]], grad_fn=<MaskedFillBackward0>)
```


```python
# understand the masked_fill operation
print('understanding masked_fill with -inf')
print()
print('wei = wei.masked_fill(tril == 0, float(\'-inf\'))')
print()
print('step 1: tril == 0 creates a boolean mask')
print('        True where we CANNOT look (future positions)')
print('        False where we CAN look (current and past)')
print()
print('step 2: masked_fill replaces True positions with -inf')
print()
print('examining batch 0, position 0')
print(f'   before masking: wei[0, 0] could see all positions')
print(f'   after masking:  wei[0, 0] = {wei[0, 0].tolist()}')
print('   positions 1-7 are now -inf (cannot look at future)')
print()
print('examining batch 0, position 3')
print(f'   wei[0, 3] = {wei[0, 3].tolist()}')
print('   positions 0-3 have real values, positions 4-7 are -inf')
print()
print('why -inf?')
print('   because e^(-inf) = 0')
print('   when we apply softmax, -inf values become 0 probability')
print('   this completely blocks information from future positions')
```


**Output:**
```
understanding masked_fill with -inf

wei = wei.masked_fill(tril == 0, float('-inf'))

step 1: tril == 0 creates a boolean mask
        True where we CANNOT look (future positions)
        False where we CAN look (current and past)

step 2: masked_fill replaces True positions with -inf

examining batch 0, position 0
   before masking: wei[0, 0] could see all positions
   after masking:  wei[0, 0] = [-0.33323800563812256, -inf, -inf, -inf, -inf, -inf, -inf, -inf]
   positions 1-7 are now -inf (cannot look at future)

examining batch 0, position 3
   wei[0, 3] = [-0.10006709396839142, 0.8648553490638733, -0.033506155014038086, 1.0221478939056396, -inf, -inf, -inf, -inf]
   positions 0-3 have real values, positions 4-7 are -inf

why -inf?
   because e^(-inf) = 0
   when we apply softmax, -inf values become 0 probability
   this completely blocks information from future positions

```


### Step 5: Softmax to Get Probabilities


```python
# normalize the attention scores to probabilities
wei = F.softmax(wei, dim=-1)
wei
```


**Output:**
```
tensor([[[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [1.9052e-01, 8.0948e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [3.7418e-01, 5.6820e-02, 5.6900e-01, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [1.2878e-01, 3.3800e-01, 1.3765e-01, 3.9557e-01, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [4.3106e-01, 8.4134e-02, 5.8193e-02, 3.0487e-01, 1.2175e-01,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [5.3737e-02, 3.2051e-01, 6.9361e-02, 2.4035e-01, 2.5680e-01,
          5.9238e-02, 0.0000e+00, 0.0000e+00],
         [3.3957e-01, 1.4859e-02, 5.1650e-01, 1.7974e-02, 6.5784e-02,
          8.0182e-03, 3.7289e-02, 0.0000e+00],
         [1.6459e-02, 3.7486e-02, 1.4404e-02, 1.1200e-01, 3.3159e-02,
          4.0686e-01, 3.1364e-01, 6.5997e-02]],

        [[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [5.7619e-01, 4.2381e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [3.3523e-01, 4.4705e-02, 6.2006e-01, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [9.8771e-01, 3.2071e-04, 7.1268e-03, 4.8401e-03, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [6.1209e-04, 9.5163e-01, 1.1413e-02, 1.5912e-02, 2.0434e-02,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [4.0501e-04, 7.5514e-01, 2.6720e-02, 1.1563e-01, 5.4259e-02,
          4.7846e-02, 0.0000e+00, 0.0000e+00],
         [5.8107e-01, 2.3946e-02, 5.3307e-02, 5.5501e-02, 9.0190e-02,
          5.5652e-02, 1.4033e-01, 0.0000e+00],
         [3.0533e-03, 7.5160e-01, 1.7858e-02, 6.1689e-02, 1.1131e-02,
          3.2726e-02, 6.1221e-02, 6.0719e-02]],

        [[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [8.4403e-02, 9.1560e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [5.4919e-02, 3.2512e-01, 6.1996e-01, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [8.0239e-02, 2.0417e-01, 6.3460e-01, 8.0984e-02, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [4.6332e-01, 4.4701e-02, 2.5986e-02, 3.3167e-01, 1.3432e-01,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [3.0988e-03, 6.6612e-03, 1.8633e-02, 1.3084e-03, 9.3273e-03,
          9.6097e-01, 0.0000e+00, 0.0000e+00],
         [1.8549e-01, 3.0068e-02, 2.9878e-02, 1.2412e-02, 1.9312e-01,
          3.7283e-02, 5.1176e-01, 0.0000e+00],
         [4.5383e-03, 6.0481e-03, 5.5622e-03, 1.2949e-01, 6.1569e-01,
          1.7967e-03, 2.0416e-01, 3.2725e-02]],

        [[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [5.3170e-01, 4.6830e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [5.5138e-02, 2.7724e-01, 6.6763e-01, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [4.1039e-01, 1.1268e-01, 5.9834e-02, 4.1710e-01, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [5.4961e-01, 1.5550e-01, 1.0927e-02, 9.3633e-02, 1.9033e-01,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [2.0316e-01, 8.0623e-02, 4.4527e-02, 3.9438e-01, 1.2515e-01,
          1.5217e-01, 0.0000e+00, 0.0000e+00],
         [8.0151e-01, 3.1811e-02, 1.6882e-03, 3.8008e-02, 2.1936e-02,
          1.0068e-01, 4.3691e-03, 0.0000e+00],
         [1.1754e-01, 2.9599e-01, 1.1695e-01, 5.4438e-02, 1.1391e-01,
          5.7793e-02, 4.5152e-02, 1.9822e-01]]], grad_fn=<SoftmaxBackward0>)
```


```python
# understand softmax normalization
print('understanding softmax normalization')
print()
print('wei = F.softmax(wei, dim=-1)')
print()
print('softmax converts raw scores to probabilities')
print('   - all values become positive (e^x > 0)')
print('   - all values in a row sum to 1')
print('   - e^(-inf) = 0 (masked positions become 0)')
print()
print('examining batch 0, position 0')
print(f'   wei[0, 0] = {[round(v, 4) for v in wei[0, 0].tolist()]}')
print(f'   sum = {wei[0, 0].sum().item():.4f}')
print('   only position 0 is visible, so it gets weight 1.0')
print()
print('examining batch 0, position 3')
print(f'   wei[0, 3] = {[round(v, 4) for v in wei[0, 3].tolist()]}')
print(f'   sum = {wei[0, 3].sum().item():.4f}')
print('   positions 0-3 have non-zero weights that sum to 1')
print()
print('this is DATA-DEPENDENT attention!')
print('the weights are NOT uniform (like 0.25, 0.25, 0.25, 0.25)')
print('instead, they depend on the query-key dot products')
```


**Output:**
```
understanding softmax normalization

wei = F.softmax(wei, dim=-1)

softmax converts raw scores to probabilities
   - all values become positive (e^x > 0)
   - all values in a row sum to 1
   - e^(-inf) = 0 (masked positions become 0)

examining batch 0, position 0
   wei[0, 0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
   sum = 1.0000
   only position 0 is visible, so it gets weight 1.0

examining batch 0, position 3
   wei[0, 3] = [0.1288, 0.338, 0.1376, 0.3956, 0.0, 0.0, 0.0, 0.0]
   sum = 1.0000
   positions 0-3 have non-zero weights that sum to 1

this is DATA-DEPENDENT attention!
the weights are NOT uniform (like 0.25, 0.25, 0.25, 0.25)
instead, they depend on the query-key dot products

```


### Step 6: Compute Values & Weighted Sum


```python
# compute values
v = value(x)  # (B, T, head_size)
v
```


**Output:**
```
tensor([[[ 0.7630, -0.2412, -0.4150,  0.3833,  0.5740, -1.6738,  0.7954,
           0.6872, -0.3848,  0.5073, -0.5312, -0.1221,  0.0445,  1.2169,
           0.9940,  1.5281],
         [ 0.3218, -0.0569, -0.8477, -0.7261,  0.0893, -0.1100, -0.0939,
          -1.0305,  0.0200,  0.2691,  0.5359,  0.1426,  0.1681,  0.3577,
           0.2332, -0.5051],
         [-0.1803,  0.2362,  0.1637,  0.5017,  0.7742, -0.4373, -0.1290,
          -0.2560, -0.4637,  0.2134,  0.0475, -0.3715, -0.8919,  0.1818,
           1.0010,  0.8075],
         [-0.3670,  0.6613,  0.3709,  0.2785, -0.5886, -0.1584, -0.5308,
          -0.9533, -0.2020,  0.1104,  0.0643,  0.5399,  0.8794,  0.3147,
          -0.3706, -0.2829],
         [-0.2686,  0.1055, -0.7112,  1.0555,  0.3917, -0.2944, -1.1679,
           0.3537,  0.7702, -0.4535, -0.7852, -0.2078, -0.5898, -0.0895,
           0.1527, -0.2386],
         [-0.3207, -0.1223,  0.2610, -0.5424, -0.0771, -0.4658,  0.2526,
          -0.0036, -0.5229,  0.1433,  0.9100,  0.2990,  0.4837,  0.2490,
          -0.0583, -0.0475],
         [ 0.5571, -0.0288, -0.2621, -0.6581, -0.5829, -0.7672,  0.0551,
          -0.7210, -0.0195, -0.4331, -0.6869, -1.0376, -0.0704,  0.3517,
          -0.4246,  0.4856],
         [-0.1362, -0.4772, -0.0267, -0.5559, -0.2265,  0.1733,  0.3853,
           0.6409, -0.2373, -0.0300,  0.1528,  0.1062,  0.1487,  0.1704,
           0.5757,  0.6982]],

        [[-1.0970,  0.4693, -1.4577,  0.1591, -0.5800,  0.4218, -0.2913,
          -0.4384, -0.6460, -0.3378, -0.5066,  1.4041,  0.3326, -0.3679,
           0.0863,  0.9563],
         [ 0.0487,  0.0677,  0.8809,  0.6150,  0.0918,  0.0283,  0.0851,
          -0.1105,  0.8593,  0.1873, -0.1565, -1.3734, -0.3197, -0.5941,
          -0.4458,  0.1768],
         [ 0.6274,  0.3655, -0.3945,  0.1603, -0.4681, -0.1591,  0.1339,
           0.1754, -0.0278, -0.0900,  0.1749,  0.2202, -0.1974,  1.3567,
           0.1852, -0.1852],
         [ 0.1319, -0.7497, -0.1804, -0.6604,  0.3357, -0.4459,  0.3611,
           0.0562,  1.0438,  0.5156,  0.6546, -0.8129, -0.4794, -0.4133,
           0.4109, -0.0929],
         [-0.4059,  0.1516, -0.1648,  0.6007,  0.2016,  0.6715,  0.2267,
          -0.3642, -0.9720,  0.3649, -0.5970, -0.0378, -0.2092,  0.2979,
           1.2460,  0.0566],
         [-0.6157, -0.6624, -0.3937,  0.1310,  0.4711,  0.8800,  0.2762,
           0.7958,  0.4129,  0.7078, -0.0498,  1.0576,  0.3581, -0.2541,
           0.6451,  0.3365],
         [ 0.8644,  0.4706, -0.1293, -0.6170, -0.5255,  0.4717,  0.8747,
          -1.1380,  0.0565,  0.7913,  0.8005, -0.2364,  1.0347, -1.0927,
          -0.1297,  0.2886],
         [ 0.9694, -0.2682,  0.1122, -0.4056,  0.7449,  0.2006,  0.4392,
           0.9176,  0.3531,  0.7124,  0.8635, -1.1062, -0.7719,  0.8359,
           0.3966, -0.0352]],

        [[ 0.3106,  0.9371, -0.2092,  0.7705,  0.0193,  0.5726,  0.2484,
          -0.9781,  0.6597,  0.5011,  0.7583, -0.0264,  0.7907,  0.1997,
          -0.4586, -0.6579],
         [ 0.0273,  0.3817,  0.6696, -0.6766,  0.0330, -0.3337, -0.5152,
          -1.1344, -0.6060,  0.4816, -0.1285, -0.0487,  0.3002, -1.7130,
          -0.8980, -0.8737],
         [-0.5203, -0.0333,  0.4968, -0.5467, -0.2312,  0.2681,  0.2471,
          -0.1932,  0.3291, -0.0638,  0.4587, -0.4808, -0.4177, -0.5063,
           0.2593,  0.0968],
         [ 0.2236,  0.1102, -0.3086, -0.4969, -0.5499, -0.7894, -0.3272,
          -0.7428, -0.8734, -0.0728,  0.1639,  0.1849,  0.8782, -0.4819,
           0.0294,  0.7049],
         [ 0.0931,  0.3978, -0.1147, -0.2994,  0.4704,  0.0658,  0.4794,
           0.0721, -0.4440, -0.1687,  1.2164, -0.1396, -1.5568,  0.2090,
           0.9272, -0.4725],
         [-0.2449, -1.0554, -0.6655,  0.4118, -0.3106,  0.9528, -0.1242,
           0.7541,  1.5634,  0.6041, -0.4255,  0.1662,  0.5790, -0.1300,
           0.5525,  0.5753],
         [ 0.1237, -0.1306, -0.1187,  0.1009, -0.3313,  0.2688, -0.8307,
           0.3135, -0.3795, -0.3788,  0.0449,  0.1885,  0.5644,  0.4173,
           0.1475, -0.7490],
         [ 0.1074,  0.4722,  0.4970,  0.7408,  0.9470, -0.0367, -0.7831,
          -0.5337, -0.5380,  0.2420, -0.7910,  0.7542, -0.0514,  0.7723,
          -0.7424, -0.7498]],

        [[ 0.0771, -0.1557, -0.9221, -0.3849,  0.4312,  0.7572,  0.2792,
           0.5681,  1.3448,  0.3264,  0.1471,  0.4326, -1.2313,  0.6333,
          -0.2560, -1.0371],
         [-0.6234, -0.1336, -1.6580,  0.5568,  0.8131, -0.3600,  0.6075,
           0.4096,  0.7376, -0.1125,  0.0733, -0.1014, -0.2114,  0.8178,
           1.1383,  0.8535],
         [ 0.2804, -0.0703,  0.5089, -0.2839,  0.6286,  0.4214,  0.1597,
          -0.3707, -0.1932,  0.7346,  0.5754, -1.1367, -0.2369, -0.3359,
           0.6683,  0.1314],
         [-1.1991,  0.0615, -0.8441, -0.5543,  0.3382,  0.0924,  0.6773,
          -0.1560, -0.7958, -0.1810, -0.5697,  0.9682, -0.7534,  0.3174,
           0.2770,  0.0683],
         [-0.3574,  0.1819,  0.5249,  0.2009, -0.9431,  0.5530, -0.6007,
           0.2637,  0.4426, -0.8211,  0.3333, -0.9226,  0.0535, -0.0387,
          -0.6463,  0.3352],
         [-0.1651, -0.0289, -0.4991, -0.4394,  0.0470,  0.8903,  0.6546,
           0.2552,  0.1525,  0.4128, -0.2636, -0.0825, -0.3186, -0.3558,
           0.4890, -0.0732],
         [ 0.4685,  0.3434, -0.8854,  0.8242, -0.2039, -0.5476,  0.0932,
           0.3258,  0.5255, -1.1008,  0.7351,  0.0765, -0.1057,  0.6164,
          -0.4499,  0.7024],
         [-0.0696,  0.0759,  0.5105,  0.2474,  0.0713,  0.0825,  0.2319,
          -0.0218, -0.1033,  0.4329, -0.0841, -0.6620,  0.1741,  0.2497,
           0.1832,  0.6219]]], grad_fn=<UnsafeViewBackward0>)
```


```python
# understand values
print('understanding values (v)')
print()
print('v = value(x) projects each token to a "value" representation')
print(f'v shape: {v.shape} = (B={B}, T={T}, head_size={head_size})')
print()
print('what does each value vector represent?')
print('   it\'s the "content" that each token will contribute')
print('   if position j gets high attention weight, its value contributes more')
print()
print('for batch 0, position 0')
print(f'   v[0, 0] = {v[0, 0].tolist()[:4]}... (first 4 of {head_size} values)')
print('   this is what position 0 will "give" when attended to')
print()
print('for batch 0, position 1')
print(f'   v[0, 1] = {v[0, 1].tolist()[:4]}... (first 4 of {head_size} values)')
print('   this is what position 1 will "give" when attended to')
```


**Output:**
```
understanding values (v)

v = value(x) projects each token to a "value" representation
v shape: torch.Size([4, 8, 16]) = (B=4, T=8, head_size=16)

what does each value vector represent?
   it's the "content" that each token will contribute
   if position j gets high attention weight, its value contributes more

for batch 0, position 0
   v[0, 0] = [0.7629690766334534, -0.24118372797966003, -0.4150242507457733, 0.3832956552505493]... (first 4 of 16 values)
   this is what position 0 will "give" when attended to

for batch 0, position 1
   v[0, 1] = [0.3217601180076599, -0.0569150447845459, -0.8477029800415039, -0.7260561585426331]... (first 4 of 16 values)
   this is what position 1 will "give" when attended to

```


```python
# compute the weighted sum of the values
out = wei @ v  # (B, T, T) @ (B, T, 16) → (B, T, 16)
out
```


**Output:**
```
tensor([[[ 7.6297e-01, -2.4118e-01, -4.1502e-01,  3.8330e-01,  5.7404e-01,
          -1.6738e+00,  7.9543e-01,  6.8724e-01, -3.8477e-01,  5.0733e-01,
          -5.3124e-01, -1.2214e-01,  4.4479e-02,  1.2169e+00,  9.9396e-01,
           1.5281e+00],
         [ 4.0582e-01, -9.2022e-02, -7.6527e-01, -5.1470e-01,  1.8168e-01,
          -4.0795e-01,  7.5564e-02, -7.0327e-01, -5.7126e-02,  3.1450e-01,
           3.3258e-01,  9.2198e-02,  1.4456e-01,  5.2138e-01,  3.7812e-01,
          -1.1776e-01],
         [ 2.0116e-01,  4.0931e-02, -1.1034e-01,  3.8763e-01,  6.6036e-01,
          -8.8138e-01,  2.1889e-01,  5.2931e-02, -4.0671e-01,  3.2654e-01,
          -1.4131e-01, -2.4897e-01, -4.8128e-01,  5.7909e-01,  9.5475e-01,
           1.0026e+00],
         [ 3.7002e-02,  2.4381e-01, -1.7073e-01, -1.6806e-02, -2.2144e-02,
          -3.7560e-01, -1.5701e-01, -6.7214e-01, -1.8652e-01,  2.2933e-01,
           1.4469e-01,  1.9491e-01,  2.8767e-01,  4.2711e-01,  1.9802e-01,
           2.5314e-02],
         [ 2.0086e-01,  1.1945e-01, -2.1421e-01,  3.4675e-01,  1.6826e-01,
          -8.4036e-01,  2.3462e-02, -5.2923e-02, -1.5896e-01,  2.3218e-01,
          -2.5714e-01,  7.7031e-02,  1.7773e-01,  6.5026e-01,  4.1195e-01,
           5.4791e-01],
         [-4.4578e-02,  1.6398e-01, -3.6067e-01,  1.2856e-01,  6.7733e-02,
          -2.9680e-01, -4.0882e-01, -4.4963e-01,  7.1842e-02,  4.6864e-02,
           1.4222e-02,  1.0751e-01,  8.2987e-02,  2.6006e-01,  1.4429e-01,
          -1.5585e-01],
         [ 1.6465e-01,  5.6039e-02, -1.1679e-01,  4.2405e-01,  5.8894e-01,
          -8.5044e-01,  1.1978e-01,  6.5047e-02, -3.2777e-01,  2.4363e-01,
          -2.1672e-01, -2.7148e-01, -4.6480e-01,  5.2730e-01,  8.4511e-01,
           9.2544e-01],
         [ 7.2800e-03, -1.5442e-02,  3.9108e-03, -4.1127e-01, -2.5813e-01,
          -4.8418e-01,  5.5030e-02, -3.1134e-01, -2.4386e-01, -6.0700e-02,
           1.5806e-01, -1.4520e-01,  2.5764e-01,  2.9121e-01, -1.1582e-01,
           1.5733e-01]],

        [[-1.0970e+00,  4.6927e-01, -1.4577e+00,  1.5908e-01, -5.8003e-01,
           4.2184e-01, -2.9130e-01, -4.3844e-01, -6.4602e-01, -3.3779e-01,
          -5.0662e-01,  1.4041e+00,  3.3257e-01, -3.6793e-01,  8.6313e-02,
           9.5632e-01],
         [-6.1147e-01,  2.9910e-01, -4.6657e-01,  3.5230e-01, -2.9529e-01,
           2.5505e-01, -1.3177e-01, -2.9946e-01, -8.0582e-03, -1.1524e-01,
          -3.5823e-01,  2.2693e-01,  5.6150e-02, -4.6380e-01, -1.3922e-01,
           6.2596e-01],
         [ 2.3433e-02,  3.8697e-01, -6.9387e-01,  1.8022e-01, -4.8061e-01,
           4.4059e-02, -1.0833e-02, -4.3189e-02, -1.9537e-01, -1.6067e-01,
          -6.8359e-02,  5.4580e-01, -2.5206e-02,  6.9131e-01,  1.2385e-01,
           2.1368e-01],
         [-1.0784e+00,  4.6250e-01, -1.4431e+00,  1.5527e-01, -5.7458e-01,
           4.1337e-01, -2.8499e-01, -4.3156e-01, -6.3295e-01, -3.3172e-01,
          -4.9603e-01,  1.3840e+00,  3.2465e-01, -3.5593e-01,  8.8418e-02,
           9.4286e-01],
         [ 4.6625e-02,  6.0098e-02,  8.2663e-01,  5.8893e-01,  9.1147e-02,
           3.1998e-02,  9.2721e-02, -1.0999e-01,  8.1375e-01,  1.9270e-01,
          -1.4900e-01, -1.3173e+00, -3.1815e-01, -5.5063e-01, -3.9011e-01,
           1.6641e-01],
         [ 1.6852e-02, -4.9043e-02,  6.0541e-01,  4.3124e-01,  1.2890e-01,
           4.4272e-02,  1.3500e-01, -5.4134e-02,  7.3558e-01,  2.5220e-01,
          -7.2767e-02, -1.0761e+00, -2.9618e-01, -4.5633e-01, -1.8570e-01,
           1.3738e-01],
         [-5.4510e-01,  2.9503e-01, -9.1187e-01,  5.3948e-02, -3.7050e-01,
           3.8831e-01,  1.8516e-02, -3.9320e-01, -3.5511e-01,  1.5370e-02,
          -1.9676e-01,  7.7186e-01,  2.9472e-01, -3.1925e-01,  2.0224e-01,
           6.0923e-01],
         [ 1.3970e-01,  5.1662e-03,  6.2362e-01,  3.7341e-01,  1.1032e-01,
           6.9541e-02,  1.7953e-01, -6.9769e-02,  7.3536e-01,  2.8889e-01,
           1.7516e-02, -1.1217e+00, -2.4648e-01, -4.7009e-01, -2.5505e-01,
           1.5395e-01]],

        [[ 3.1063e-01,  9.3706e-01, -2.0925e-01,  7.7045e-01,  1.9348e-02,
           5.7261e-01,  2.4843e-01, -9.7810e-01,  6.5967e-01,  5.0115e-01,
           7.5831e-01, -2.6434e-02,  7.9072e-01,  1.9970e-01, -4.5862e-01,
          -6.5785e-01],
         [ 5.1229e-02,  4.2860e-01,  5.9542e-01, -5.5449e-01,  3.1818e-02,
          -2.5724e-01, -4.5078e-01, -1.1213e+00, -4.9919e-01,  4.8322e-01,
          -5.3645e-02, -4.6865e-02,  3.4156e-01, -1.5516e+00, -8.6092e-01,
          -8.5549e-01],
         [-2.9662e-01,  1.5494e-01,  5.1421e-01, -5.1660e-01, -1.3153e-01,
           8.9164e-02, -6.5588e-04, -5.4235e-01,  4.3208e-02,  1.4455e-01,
           2.8422e-01, -3.1536e-01, -1.1797e-01, -8.5984e-01, -1.5641e-01,
          -2.6020e-01],
         [-2.8156e-01,  1.4094e-01,  4.1021e-01, -4.6351e-01, -1.8295e-01,
           8.4020e-02,  4.5066e-02, -4.9289e-01,  6.7285e-02,  9.2170e-02,
           3.3895e-01, -3.0220e-01, -6.9244e-02, -6.9405e-01, -5.3239e-02,
          -1.1269e-01],
         [ 2.1831e-01,  5.4036e-01, -1.7186e-01,  1.0750e-01, -1.1477e-01,
           4.3744e-03,  5.4357e-02, -7.4557e-01, -6.2228e-02,  2.0526e-01,
           5.7527e-01,  1.5657e-02,  4.5107e-01, -1.2896e-01, -1.1162e-01,
          -1.7100e-01],
         [-2.4276e-01, -1.0055e+00, -6.2789e-01,  3.8000e-01, -2.9882e-01,
           9.1979e-01, -1.1335e-01,  7.1017e-01,  1.5012e+00,  5.8242e-01,
          -3.8730e-01,  1.4929e-01,  5.3973e-01, -1.4386e-01,  5.3706e-01,
           5.4326e-01],
         [ 1.1782e-01,  1.5628e-01, -1.1536e-01,  1.0920e-01, -9.9444e-02,
           2.8018e-01, -3.0325e-01, -2.8041e-02, -1.1854e-01, -9.9304e-02,
           3.9456e-01,  5.7266e-02,  1.6387e-01,  2.1351e-01,  1.7117e-01,
          -5.8975e-01],
         [ 1.1331e-01,  2.5244e-01, -1.1388e-01, -2.0675e-01,  1.8021e-01,
          -4.2304e-03,  5.6730e-02, -1.6245e-02, -4.7757e-01, -1.7682e-01,
           7.5786e-01, -1.6106e-03, -7.2714e-01,  1.6426e-01,  5.7538e-01,
          -3.8376e-01]],

        [[ 7.7075e-02, -1.5568e-01, -9.2207e-01, -3.8491e-01,  4.3124e-01,
           7.5718e-01,  2.7922e-01,  5.6814e-01,  1.3448e+00,  3.2643e-01,
           1.4705e-01,  4.3255e-01, -1.2313e+00,  6.3330e-01, -2.5603e-01,
          -1.0371e+00],
         [-2.5095e-01, -1.4535e-01, -1.2667e+00,  5.6085e-02,  6.1009e-01,
           2.3402e-01,  4.3293e-01,  4.9392e-01,  1.0604e+00,  1.2088e-01,
           1.1250e-01,  1.8252e-01, -7.5366e-01,  7.1970e-01,  3.9696e-01,
          -1.5175e-01],
         [ 1.8618e-02, -9.2575e-02, -1.7076e-01, -5.6385e-02,  6.6891e-01,
           2.2329e-01,  2.9041e-01, -1.0262e-01,  1.4965e-01,  4.7727e-01,
           4.1259e-01, -7.6315e-01, -2.8469e-01,  3.7417e-02,  7.4762e-01,
           2.6716e-01],
         [-5.2196e-01, -5.7511e-02, -8.8683e-01, -3.4342e-01,  4.4727e-01,
           3.3393e-01,  4.7508e-01,  1.9206e-01,  2.9150e-01,  8.9754e-02,
          -1.3457e-01,  5.0191e-01, -8.5755e-01,  4.6433e-01,  1.7871e-01,
          -2.9310e-01],
         [-2.3181e-01, -6.6739e-02, -7.3817e-01, -1.4173e-01,  2.2250e-01,
           4.7868e-01,  1.9876e-01,  4.0749e-01,  8.6140e-01, -3.2872e-03,
           1.0859e-01,  1.2460e-01, -7.7254e-01,  4.9392e-01, -5.3462e-02,
          -3.6565e-01],
         [-5.6486e-01, -2.9277e-03, -6.4147e-01, -3.0628e-01,  2.0365e-01,
           3.8468e-01,  4.0435e-01,  1.4225e-01,  8.8813e-02, -2.1360e-02,
          -1.6166e-01,  2.8291e-01, -6.1666e-01,  2.4582e-01,  1.7229e-01,
          -7.8285e-02],
         [-2.5574e-02, -1.2424e-01, -8.6561e-01, -3.4857e-01,  3.6858e-01,
           6.9903e-01,  3.2227e-01,  4.9475e-01,  1.0981e+00,  2.7116e-01,
           8.3502e-02,  3.5014e-01, -1.0540e+00,  5.1113e-01, -1.2425e-01,
          -7.9824e-01],
         [-2.5083e-01, -1.3141e-02, -4.9340e-01,  1.3992e-01,  2.8353e-01,
           1.4285e-01,  2.8776e-01,  1.9136e-01,  3.7297e-01,  4.7570e-02,
           1.1451e-01, -2.9702e-01, -2.5862e-01,  3.4686e-01,  3.7072e-01,
           3.3875e-01]]], grad_fn=<UnsafeViewBackward0>)
```


```python
# understand the final output
print('understanding the weighted sum: out = wei @ v')
print()
print(f'wei shape: {wei.shape} = (B={B}, T={T}, T={T})')
print(f'v shape:   {v.shape} = (B={B}, T={T}, head_size={head_size})')
print(f'out shape: {out.shape} = (B={B}, T={T}, head_size={head_size})')
print()
print('what does out[b, i] represent?')
print('   it\'s a weighted average of all value vectors')
print('   weighted by how much position i attends to each position')
print()
print('for position 0 (can only see itself)')
print(f'   wei[0, 0] = {[round(v, 3) for v in wei[0, 0].tolist()]}')
print('   out[0, 0] = 1.0 * v[0, 0] + 0.0 * v[0, 1] + ... + 0.0 * v[0, 7]')
print('   out[0, 0] ≈ v[0, 0] (just itself)')
print()
print('for position 3 (can see positions 0, 1, 2, 3)')
weights_3 = [round(w, 3) for w in wei[0, 3].tolist()]
print(f'   wei[0, 3] = {weights_3}')
print(f'   out[0, 3] = {weights_3[0]} * v[0, 0] + {weights_3[1]} * v[0, 1] + {weights_3[2]} * v[0, 2] + {weights_3[3]} * v[0, 3]')
print('   out[0, 3] is a weighted mix of values from positions 0-3')
```


**Output:**
```
understanding the weighted sum: out = wei @ v

wei shape: torch.Size([4, 8, 8]) = (B=4, T=8, T=8)
v shape:   torch.Size([4, 8, 16]) = (B=4, T=8, head_size=16)
out shape: torch.Size([4, 8, 16]) = (B=4, T=8, head_size=16)

what does out[b, i] represent?
   it's a weighted average of all value vectors
   weighted by how much position i attends to each position

for position 0 (can only see itself)
   wei[0, 0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
   out[0, 0] = 1.0 * v[0, 0] + 0.0 * v[0, 1] + ... + 0.0 * v[0, 7]
   out[0, 0] ≈ v[0, 0] (just itself)

for position 3 (can see positions 0, 1, 2, 3)
   wei[0, 3] = [0.129, 0.338, 0.138, 0.396, 0.0, 0.0, 0.0, 0.0]
   out[0, 3] = 0.129 * v[0, 0] + 0.338 * v[0, 1] + 0.138 * v[0, 2] + 0.396 * v[0, 3]
   out[0, 3] is a weighted mix of values from positions 0-3

```


```python
# manually verify one output calculation
print('manually verifying out[0, 0]')
print()
print('out[0, 0] should equal v[0, 0] since position 0 only sees itself')
print()
print(f'out[0, 0] (first 4 values): {out[0, 0].tolist()[:4]}')
print(f'v[0, 0] (first 4 values):   {v[0, 0].tolist()[:4]}')
print()
print(f'are they equal? {torch.allclose(out[0, 0], v[0, 0])}')
```


**Output:**
```
manually verifying out[0, 0]

out[0, 0] should equal v[0, 0] since position 0 only sees itself

out[0, 0] (first 4 values): [0.7629690766334534, -0.24118372797966003, -0.4150242507457733, 0.3832956552505493]
v[0, 0] (first 4 values):   [0.7629690766334534, -0.24118372797966003, -0.4150242507457733, 0.3832956552505493]

are they equal? True

```


```python
# manually verify position 1 calculation
print('manually verifying out[0, 1]')
print()
print('position 1 can see positions 0 and 1')
print(f'wei[0, 1] = {[round(w, 4) for w in wei[0, 1].tolist()]}')
print()
print('computing the weighted sum manually')
w0 = wei[0, 1, 0].item()
w1 = wei[0, 1, 1].item()
print(f'   weight for position 0: {w0:.4f}')
print(f'   weight for position 1: {w1:.4f}')
print()
manual_out = w0 * v[0, 0] + w1 * v[0, 1]
print(f'manual calculation:')
print(f'   {w0:.4f} * v[0, 0] + {w1:.4f} * v[0, 1]')
print(f'   = {manual_out.tolist()[:4]}... (first 4 values)')
print()
print(f'actual out[0, 1]:')
print(f'   = {out[0, 1].tolist()[:4]}... (first 4 values)')
print()
print(f'are they equal? {torch.allclose(manual_out, out[0, 1])}')
```


**Output:**
```
manually verifying out[0, 1]

position 1 can see positions 0 and 1
wei[0, 1] = [0.1905, 0.8095, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

computing the weighted sum manually
   weight for position 0: 0.1905
   weight for position 1: 0.8095

manual calculation:
   0.1905 * v[0, 0] + 0.8095 * v[0, 1]
   = [0.4058188199996948, -0.09202173352241516, -0.7652695178985596, -0.5147035717964172]... (first 4 values)

actual out[0, 1]:
   = [0.4058188199996948, -0.09202173352241516, -0.7652694582939148, -0.5147035717964172]... (first 4 values)

are they equal? True

```


### Step 7: Visualize the Attention Weights
Let's look at what the attention pattern looks like. Remember that these are now **data-dependent** weights, not uniform averages!

#### Key Takeaways About Attention
| Concept | Explanation |
|---------|-------------|
| **Communication Mechanism** | Tokens can "talk" to each other and share information. |
| **No Position Awareness** | Attention is set-based - we need positional encodings. |
| **Batch Independence** | Sequences in a batch don't interact. |
| **Decoder vs Encoder** | Decoder uses causal mask (triangle); Encoder allows full attention. |
| **Self vs Cross Attention** | Self: Q, K, V from same source; Cross: Q from one source, K/V from another. |


```python
print('Attention Weights for First Sequence (batch 0)')
print('   Rows = \'from\' position (query)')
print('   Cols = \'to\' position (key)')
print()
print('        ', end='')
for j in range(T):
    print(f'  pos{j} ', end='')
print()
print('       ' + '-' * 55)
for i in range(T):
    print(f'   pos{i} |', end='')
    for j in range(T):
        val = wei[0, i, j].item()
        if val > 0.001:
            print(f' {val:.3f}', end=' ')
        else:
            print(f'  ---  ', end='')
    print()
print()
print('Observations')
print('   - Lower triangular (can\'t attend to future).')
print('   - Each row sums to 1 (valid probability distribution).')
print('   - Values are NOT uniform - they\'re learned!')
print('   - Different positions attend differently to past tokens.')
```


**Output:**
```
Attention Weights for First Sequence (batch 0)
   Rows = 'from' position (query)
   Cols = 'to' position (key)

          pos0   pos1   pos2   pos3   pos4   pos5   pos6   pos7 
       -------------------------------------------------------
   pos0 | 1.000   ---    ---    ---    ---    ---    ---    ---  
   pos1 | 0.191  0.809   ---    ---    ---    ---    ---    ---  
   pos2 | 0.374  0.057  0.569   ---    ---    ---    ---    ---  
   pos3 | 0.129  0.338  0.138  0.396   ---    ---    ---    ---  
   pos4 | 0.431  0.084  0.058  0.305  0.122   ---    ---    ---  
   pos5 | 0.054  0.321  0.069  0.240  0.257  0.059   ---    ---  
   pos6 | 0.340  0.015  0.517  0.018  0.066  0.008  0.037   ---  
   pos7 | 0.016  0.037  0.014  0.112  0.033  0.407  0.314  0.066 

Observations
   - Lower triangular (can't attend to future).
   - Each row sums to 1 (valid probability distribution).
   - Values are NOT uniform - they're learned!
   - Different positions attend differently to past tokens.

```


## Step 8: Scaled Attention
There's one more trick: we **scale** the attention scores by $\frac{1}{\sqrt{d_k}}$ (where $d_k$ is the head size).

### The Problem Without Scaling
When we compute Q @ K^T, the dot product grows larger as head_size increases.

For vectors with unit variance.
- If head_size = 16, the dot product has variance ≈ 16
- If head_size = 64, the dot product has variance ≈ 64
- If head_size = 512, the dot product has variance ≈ 512

### Why is Large Variance Bad?
Large values going into softmax cause problems.
- Softmax becomes "peaky" (one position gets nearly all attention)
- Gradients become very small (vanishing gradient problem)
- Training becomes unstable

### The Solution: Scale by 1/sqrt(head_size)
By dividing by sqrt(head_size), we bring the variance back to ~1.
- variance of Q @ K^T ≈ head_size
- variance of (Q @ K^T) / sqrt(head_size) ≈ 1

This is why the formula in "Attention is All You Need" is.
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$


```python
# create random keys with unit variance
k = torch.randn(B, T, head_size)
k
```


**Output:**
```
tensor([[[-0.6744, -1.8893, -1.8424,  0.1323, -0.7929,  1.2297,  0.0777,
           1.8036, -0.3388, -0.4670, -0.4019, -1.3110,  0.0308, -0.5922,
          -1.1771,  1.7409],
         [-0.2961, -0.3474, -0.4967, -1.3010,  1.3099, -0.2666,  0.1970,
          -0.6992,  1.1396,  0.1912, -0.0095,  0.3546, -0.4238,  1.0712,
           2.7125, -0.1935],
         [ 1.7503, -0.1117, -0.8220,  0.7975, -0.7685,  1.5376, -1.7771,
          -1.0646,  1.0508,  1.3841, -1.5027, -1.0865,  2.1496, -0.9262,
          -0.8618, -0.0133],
         [ 0.9761, -0.0773, -2.1688,  1.2137, -1.8086,  0.1943,  0.6680,
          -1.1589, -0.7162, -1.0271, -1.4785,  0.0458, -0.1069,  0.3531,
           0.3302, -0.5309],
         [ 0.0363,  2.4673, -0.1655, -0.3069,  1.4189, -0.4566, -1.5976,
           0.7736, -0.6360, -0.2510,  0.7005,  1.4388, -1.0685, -0.1663,
           0.5176, -0.7325],
         [ 0.3359, -0.7604,  0.0566, -1.5039, -0.4485,  0.5257,  0.2619,
           0.7167, -0.6965,  0.8436,  1.9249, -0.3405, -0.4329,  1.3084,
           0.4293,  0.0712],
         [-1.4018,  0.5611,  1.1513,  0.6989, -0.5898, -0.1646, -0.4931,
           0.5041,  0.1377,  0.2751,  0.4683, -0.7030, -0.1796,  0.8974,
           0.0517, -0.5315],
         [ 0.4069,  0.4082, -0.4961, -0.9291, -0.1993,  0.4683,  1.0864,
          -0.4892, -0.0861,  0.6074,  0.2278, -0.6186,  1.1309, -0.1208,
           1.6047,  0.0861]],

        [[ 0.2813,  0.0870, -0.2571,  2.2180,  1.2402, -0.6573,  1.8484,
          -1.1966, -0.4539,  1.4244,  2.2692,  1.3105, -0.3179, -0.3774,
           2.2604, -0.3310],
         [-0.7194,  1.2199,  1.4356, -0.3140,  0.8979,  0.6359, -0.8476,
          -0.0931, -0.3936, -0.0248, -0.3633, -0.6941, -0.9816, -0.0556,
          -1.0469, -0.1615],
         [-0.5718, -1.5561, -0.9633, -0.4366, -0.0085,  0.0446, -0.3537,
           0.1575, -1.1567,  1.8156, -2.0921, -0.6517,  1.1426, -0.7538,
          -1.4663,  0.0802],
         [-0.6317, -0.7410,  1.8064,  0.9378, -0.3845,  0.6585,  0.7617,
          -0.6451, -3.6308, -2.1864,  0.2644, -0.5599,  1.4537, -0.2963,
          -0.4702, -1.4991],
         [ 2.2968,  1.6495,  1.3179,  0.7556,  1.2472,  0.7881,  1.5493,
          -0.6089, -2.7026, -0.6109,  1.1898, -0.4802,  2.2536,  1.1718,
           0.8793, -0.7797],
         [-0.0781, -0.3723,  0.3638,  1.2563, -0.1221,  0.1012,  0.4712,
           0.6840,  0.5099, -0.7802,  0.6629,  0.6557,  0.0585,  0.7882,
          -1.0858,  1.0520],
         [-0.3893,  1.4754, -0.1709, -2.0884,  0.7963,  0.4962,  0.6029,
          -0.5226,  1.0361,  0.5318, -0.3148,  0.0210, -0.0545, -0.8116,
          -0.2611, -0.6926],
         [ 1.5523, -2.3087, -2.1958,  0.3203,  0.7727, -0.1667, -0.0118,
          -0.1128, -0.6838, -1.2514, -0.0760,  0.3789,  0.6201, -0.0899,
           1.2097,  0.8767]],

        [[ 1.8313, -0.6159, -0.6073, -2.0597,  1.5289,  0.3379,  0.1915,
           0.1635,  0.6710, -0.4096, -0.5302,  0.2533, -0.1990,  0.6101,
          -1.4391,  1.6621],
         [ 0.3556, -1.8120,  0.4646, -0.5480, -1.0596,  0.1740,  0.3822,
          -0.1958, -0.1513,  0.6256, -0.6219, -1.0873, -1.3252,  0.3772,
          -0.0584, -1.4766],
         [-0.9860,  1.4866,  0.1471, -1.3660, -0.6709,  0.9521,  1.4749,
          -1.4756, -0.8660,  1.2781,  0.3526, -0.0750,  0.4059,  0.5351,
          -0.0688, -0.6155],
         [ 0.2696, -0.0316, -1.2757, -0.6373, -0.7616, -0.4670, -1.2028,
          -2.4588, -0.4899, -1.5937,  0.9481, -0.4265, -1.4827, -0.4504,
           0.8890, -1.1526],
         [ 0.0295, -0.5199, -0.1654, -0.2773, -0.2447, -1.9880, -1.2664,
          -0.3072,  0.8398, -0.4689,  0.2266,  0.3419,  0.5934,  1.9173,
          -0.4787, -0.0578],
         [-1.7239, -0.9909,  1.9552, -0.0653,  0.1463,  1.1357, -0.2689,
          -0.9127,  0.6866,  1.5644,  1.0132, -1.1486, -0.7916, -0.3214,
           0.5456, -1.2671],
         [ 0.5780, -0.0210, -0.1380,  0.0994, -0.1628,  0.1898, -1.2572,
           0.2571, -1.0626, -0.6326, -0.6293, -1.6768,  0.6724,  1.9889,
           0.8157, -1.4683],
         [ 1.6630, -1.4545, -0.2315,  0.5550,  0.3245,  1.4937,  0.5853,
           0.7600, -1.0136, -1.3920,  0.8857,  0.9162,  0.4851, -1.0356,
           0.1621, -0.3456]],

        [[ 0.7719,  0.0167,  0.6804, -0.1298,  0.0973,  0.7957, -2.1607,
          -0.5694, -2.0023, -1.2304,  0.8770, -2.0921,  1.5937,  2.5637,
          -0.1268,  0.2314],
         [ 0.7924, -0.3076,  0.6760,  2.6806, -0.8708,  0.0361,  1.0990,
          -0.2800,  0.5311,  0.5320, -1.5853,  2.4220,  0.4772,  0.5957,
           0.2793,  0.2393],
         [ 0.4738,  0.0311, -0.1489, -0.3652, -1.8156,  1.1129,  1.1716,
          -1.7179,  1.0240, -1.0366, -1.9978,  1.5088,  0.1965,  1.0685,
           0.4851,  0.0060],
         [ 1.0007,  0.7049, -0.6978,  0.4729, -0.6567, -0.8678, -0.1043,
           0.9756, -0.8829, -0.7063, -1.2800,  0.1359, -0.2811,  1.7253,
           0.1270, -0.8810],
         [-0.6381,  0.5337,  0.1681, -1.0806,  0.8653,  0.9823,  0.7240,
           0.1330, -0.6278, -0.1459, -0.4236, -1.4882,  0.8582,  3.0351,
          -1.1488,  0.2271],
         [ 0.0306,  0.0151,  1.1773, -0.9650, -0.2467, -0.6798, -1.0098,
          -0.3883, -1.3796,  1.0700, -0.9035,  0.7684,  0.4392, -0.5033,
           2.1168,  1.2190],
         [-0.7853,  1.0901, -0.0665,  1.2573,  0.1582, -1.7430, -1.2939,
           1.3075,  0.7086,  0.2949, -0.6938, -0.8013, -0.0776, -0.5015,
          -2.2270, -0.1726],
         [-0.6626, -0.5495,  0.0587,  1.5382,  1.0445, -0.2630,  0.2191,
           0.0512,  1.1272,  0.5445, -0.2186,  0.4121, -1.1325, -2.3891,
           0.7178, -1.5831]]])
```


```python
# Create random queries with unit variance
q = torch.randn(B, T, head_size)
q
```


**Output:**
```
tensor([[[-9.6348e-01, -1.0543e+00, -6.1099e-01,  1.1033e-01,  1.2356e-01,
          -1.4389e+00, -4.5936e-01,  7.1935e-01, -9.6226e-02, -6.8070e-01,
           7.3392e-01,  9.3939e-02,  1.0835e+00,  8.0898e-01, -9.7732e-01,
          -2.6084e-01],
         [ 9.0191e-01,  3.1770e-01,  1.5054e+00, -4.5409e-04, -8.3999e-01,
          -9.9635e-01,  1.9696e+00, -6.2411e-01,  7.8123e-01, -1.4737e+00,
           9.1280e-01, -8.1394e-01, -3.2805e-01, -1.6034e+00,  1.5658e-01,
           1.2400e+00],
         [-1.3389e+00, -1.0444e-01,  1.5695e-01, -1.5132e+00,  9.9128e-01,
           5.5732e-01, -6.7796e-01,  9.6848e-01,  8.3635e-01, -2.0765e+00,
           9.2636e-01,  1.8823e+00,  2.7995e-02, -3.6298e-01,  4.5504e-01,
           7.5949e-01],
         [-9.6253e-01,  9.5393e-01, -1.4123e+00,  8.1285e-01,  1.4346e+00,
           5.7747e-02, -8.9515e-01, -8.5902e-02, -6.0462e-01, -6.8750e-01,
           2.0560e-01, -7.1922e-01, -1.1453e+00,  8.8890e-01,  2.4767e-01,
           9.7610e-01],
         [-1.0026e+00, -8.6914e-01,  1.0349e+00,  1.1414e+00, -6.1135e-01,
           5.6699e-01, -1.5298e-01, -2.9166e-01, -1.2069e+00, -1.6842e-01,
          -1.0213e+00,  4.5474e-01,  5.6282e-02,  1.9085e-01, -2.8136e-03,
          -6.4238e-01],
         [-2.3483e-01,  1.8348e-01,  8.2710e-01,  6.8175e-01,  4.0631e-01,
           1.7062e+00,  1.1659e+00, -2.4008e-01,  2.2485e-01, -2.3762e+00,
           4.0155e-01, -2.2946e+00,  9.5440e-01, -3.8834e-01,  2.1960e+00,
           8.4125e-01],
         [-1.4890e+00,  5.8501e-01, -6.4059e-01, -1.9064e+00, -2.1498e-01,
           1.6726e-01,  8.5943e-02, -3.8008e-01, -1.3825e+00,  5.6673e-01,
          -2.2063e+00,  2.8584e-01,  2.4994e+00,  5.4578e-02, -1.1838e+00,
           8.2043e-01],
         [ 7.9914e-01,  3.4314e-01, -7.1088e-01,  4.0654e-01,  9.5622e-01,
           3.0749e-01,  3.1811e-01, -1.8298e+00,  1.8508e+00, -1.2886e+00,
           1.2673e+00, -9.6881e-01, -4.6094e-01,  8.4074e-01, -1.9394e-01,
          -1.4038e-01]],

        [[ 7.0630e-02, -6.8063e-02,  1.2693e+00,  2.2910e+00, -7.9681e-02,
          -2.1427e+00,  1.5941e+00,  2.5384e+00, -4.9620e-01,  1.5591e+00,
          -1.2024e+00, -6.3514e-01,  2.0571e-01,  1.2085e+00, -1.2975e+00,
          -1.7842e+00],
         [-1.7387e+00,  1.7332e+00,  7.3354e-01,  9.9386e-01,  8.0198e-03,
          -2.8346e-01, -1.0711e+00, -3.9535e-01,  7.2411e-01,  1.1350e+00,
           1.3726e-01, -2.8324e-01,  8.4498e-01, -1.6652e+00,  9.0134e-01,
           2.9201e-01],
         [ 9.8307e-01, -1.6503e+00, -8.4474e-01, -1.7778e+00,  1.6798e+00,
          -3.0757e-01,  1.7818e-01,  3.0598e-01,  6.4698e-01,  1.3001e+00,
           6.4310e-01, -1.3192e+00, -5.6939e-01, -1.8996e+00,  2.5208e-01,
           7.2486e-01],
         [-1.0662e-02,  3.2636e-01, -3.9913e-01, -5.4581e-01, -9.4928e-01,
          -5.8264e-01,  6.0848e-01,  1.8587e+00, -2.0494e+00,  1.2204e+00,
           1.3751e+00,  4.9697e-01, -5.0501e-01,  1.3103e+00, -8.3440e-02,
           2.1579e-01],
         [ 1.1008e-01, -4.9934e-01,  1.4105e+00,  9.0974e-01,  8.3605e-01,
          -4.5467e-01, -1.4347e-01,  1.0890e-01,  6.9148e-01,  4.3387e-01,
           1.6470e-01,  9.2992e-01,  4.3113e-01, -2.2406e+00,  3.4794e-01,
          -1.4979e+00],
         [ 2.8999e-01, -2.8325e-01, -7.4868e-01, -1.4254e+00, -3.2493e-01,
           2.4911e-01, -1.4054e+00,  7.2349e-03, -6.6038e-01, -9.2917e-01,
           1.0411e+00,  1.7303e+00, -8.9635e-01, -1.7487e+00,  2.3790e-01,
          -1.8056e+00],
         [-4.1451e-01, -1.7113e+00,  3.6452e-01, -5.8035e-01, -7.1043e-01,
          -2.7302e-01,  8.7178e-01,  2.1582e-01,  5.2196e-01,  3.4126e-01,
           1.0109e+00, -4.6918e-02,  2.9928e-01, -4.2063e-01, -1.0377e+00,
           6.3906e-01],
         [ 2.3253e-02, -1.0358e+00,  6.9172e-01, -6.7895e-01, -9.2474e-01,
          -3.1068e-01,  1.9666e-01,  6.3062e-01, -1.7229e+00, -5.2109e-01,
           6.0717e-02,  4.2114e-01,  1.0577e+00,  1.3254e+00,  1.1332e+00,
           4.2492e-01]],

        [[ 7.7359e-01,  1.4078e-01, -2.7783e-01,  1.5577e-02, -9.0613e-01,
          -6.0330e-01,  5.2168e-02, -5.9930e-01, -3.0289e+00,  2.6210e-01,
           2.5721e+00, -4.7537e-01,  3.6165e-01, -5.7933e-01, -7.2539e-01,
           1.3805e+00],
         [-1.5331e-01, -1.3848e+00,  4.5417e-01,  3.8795e-01, -6.6643e-01,
          -5.4208e-01,  2.9302e+00,  3.9807e-02, -6.9273e-01, -6.2174e-01,
           7.4563e-01, -4.6407e-01,  3.5406e-01,  5.7339e-01, -1.9254e+00,
           1.3461e+00],
         [ 1.2138e+00, -8.6570e-02, -7.5610e-01, -6.0872e-01,  6.5236e-02,
           2.0737e-01, -4.8440e-01, -4.9315e-01, -6.0584e-01, -3.8430e-01,
          -1.5004e+00,  1.4241e+00,  4.2643e-01,  1.7253e+00, -1.1768e+00,
          -5.5720e-01],
         [ 6.2589e-01, -2.1850e-02, -1.7424e+00, -7.6729e-01,  1.4304e+00,
           3.9235e-01, -1.3970e+00, -2.0021e+00,  2.3755e-01,  2.4901e+00,
           3.1266e-01, -3.8830e-01,  8.9288e-01,  7.2496e-01,  4.3352e-01,
           9.7441e-01],
         [-4.0826e-03,  7.2984e-02, -2.5910e-01,  7.9609e-01,  5.7920e-01,
          -1.3451e-01, -1.3556e+00, -3.9413e-01,  1.3683e+00,  1.2856e+00,
          -1.3335e+00,  9.4325e-01, -1.1380e+00, -1.1207e+00, -6.6216e-01,
           1.8776e+00],
         [-9.0479e-01, -1.0662e-02, -3.0602e-01, -2.0311e+00,  2.8344e-01,
           3.8604e-01, -2.3576e-01, -1.2253e+00, -7.9281e-01,  4.7013e-01,
           9.6294e-02,  1.7162e+00, -1.5964e-01, -1.7001e-01, -8.2118e-01,
           2.3731e-02],
         [ 5.2973e-01, -3.7780e-01,  1.4174e+00,  4.9107e-01,  9.1596e-01,
           1.7351e-01,  1.8969e-01, -1.5841e-01, -1.9692e+00, -2.4453e-01,
          -6.7767e-01,  4.7823e-01, -1.8906e+00,  2.9901e-01,  1.5097e+00,
          -4.0141e-01],
         [ 3.6162e-01,  4.4447e-01,  3.7175e-01,  1.7289e-01, -1.2142e+00,
           1.9442e+00,  1.1369e+00, -9.7678e-01,  9.7180e-01,  1.5757e-01,
          -1.2937e-01,  9.4592e-01,  7.4863e-01,  1.3490e+00,  1.6593e+00,
          -6.8208e-01]],

        [[ 8.5920e-01,  1.0688e+00,  1.0354e+00, -1.0889e+00, -9.8336e-01,
           1.8595e-01,  1.5627e+00, -1.3852e+00, -4.5413e-01,  6.9793e-01,
           5.4690e-01,  5.5246e-01,  2.4029e-01, -1.1295e+00, -2.9120e-01,
          -1.4457e+00],
         [ 3.2800e-01, -2.2357e-01,  6.3385e-01,  1.2713e+00,  5.2576e-01,
          -8.8591e-02,  8.3571e-02, -5.5072e-01, -4.5323e-01,  1.0592e+00,
          -1.7509e+00,  8.3122e-01, -2.0036e-01, -1.2242e+00, -1.6658e-01,
          -6.3705e-01],
         [-8.2356e-01, -3.6673e-01,  5.7609e-01, -5.4015e-01,  2.1437e+00,
          -8.4680e-01,  1.0048e+00,  9.9498e-02, -9.2466e-01,  6.7150e-01,
           4.1903e-01, -2.2128e-01,  5.4170e-01,  2.7081e-01,  6.5265e-01,
          -2.2563e-01],
         [-2.7009e-01,  1.1325e+00,  1.2827e+00,  4.0248e-01,  1.5574e+00,
           2.0379e+00, -1.1328e-01, -1.0065e+00, -3.7983e-01, -6.7844e-01,
          -1.0526e+00,  2.2322e-01,  1.0225e+00,  7.7577e-01,  6.4597e-01,
           7.4031e-01],
         [-7.4956e-01, -1.1346e+00,  4.3102e-01, -2.2315e-01,  4.0467e-02,
           3.5534e-01,  1.9383e+00, -1.2549e+00,  4.5961e-01,  1.5001e+00,
           3.8992e-01,  4.0374e-01, -4.7906e-01, -5.7640e-01, -2.3239e+00,
          -2.8545e-01],
         [-5.0370e-01,  5.8248e-01, -2.6750e+00,  1.8529e-01, -1.3125e+00,
          -7.7555e-01, -9.4621e-02, -1.1716e+00,  5.1212e-01,  3.4507e-01,
          -1.1712e+00,  2.5597e-01,  4.5153e-01, -7.7741e-01, -2.5793e+00,
           1.3328e+00],
         [ 3.1002e-01,  8.0718e-02,  1.3723e-01,  1.2757e+00,  4.2629e-01,
           1.2246e-01,  5.5749e-01,  3.0464e-01, -5.0844e-01, -3.8419e-01,
          -1.8623e-01, -1.6419e-01,  5.0378e-01,  1.2022e+00,  5.4761e-01,
          -5.0970e-01],
         [ 1.6420e+00, -1.3696e+00, -1.3332e-01,  6.1232e-01, -2.7727e+00,
           4.3604e-01,  1.2461e-01,  1.5180e+00,  6.8525e-01,  1.6848e+00,
          -5.5461e-01, -3.9066e-01,  1.0325e+00,  1.6486e-01, -5.9153e-01,
          -1.4602e-02]]])
```


```python
# understand the scaling problem
print('understanding the scaling problem')
print()
print(f'head_size = {head_size}')
print()
print('we created random keys and queries with unit variance (~1)')
print(f'   key variance:   {k.var().item():.4f} (should be ~1)')
print(f'   query variance: {q.var().item():.4f} (should be ~1)')
print()
print('when we multiply unit-variance vectors, variance grows')
print('   for dot product of two vectors with unit variance')
print('   the resulting variance ≈ length of vectors = head_size')
print()
print(f'   expected variance after Q @ K^T: ~{head_size}')
```


**Output:**
```
understanding the scaling problem

head_size = 16

we created random keys and queries with unit variance (~1)
   key variance:   1.0392 (should be ~1)
   query variance: 0.9791 (should be ~1)

when we multiply unit-variance vectors, variance grows
   for dot product of two vectors with unit variance
   the resulting variance ≈ length of vectors = head_size

   expected variance after Q @ K^T: ~16

```


```python
# demonstrate the problem: unscaled attention
print('demonstrating the problem: unscaled attention')
print()
wei_unscaled = q @ k.transpose(-2, -1)
print(f'wei_unscaled = q @ k.transpose(-2, -1)')
print(f'wei_unscaled variance: {wei_unscaled.var().item():.4f}')
print()
print(f'expected variance: ~{head_size}')
print(f'actual variance:   {wei_unscaled.var().item():.2f}')
print()
print('the variance grew from ~1 to ~{head_size}!')
print('this is because dot product of two unit-variance vectors')
print(f'has variance equal to the length of the vectors ({head_size})')
```


**Output:**
```
demonstrating the problem: unscaled attention

wei_unscaled = q @ k.transpose(-2, -1)
wei_unscaled variance: 12.4662

expected variance: ~16
actual variance:   12.47

the variance grew from ~1 to ~{head_size}!
this is because dot product of two unit-variance vectors
has variance equal to the length of the vectors (16)

```


```python
# the solution: scaled attention
print('the solution: scaled attention')
print()
scale_factor = head_size ** -0.5
print(f'scale factor = 1 / sqrt(head_size) = 1 / sqrt({head_size}) = {scale_factor:.4f}')
print()
wei_scaled = (q @ k.transpose(-2, -1)) * scale_factor
print(f'wei_scaled = (q @ k.transpose(-2, -1)) * {scale_factor:.4f}')
print()
print(f'wei_scaled variance: {wei_scaled.var().item():.4f}')
print()
print('the variance is back to ~1!')
print()
print('why does this work?')
print(f'   unscaled variance ≈ {head_size}')
print(f'   multiplying by {scale_factor:.4f} = 1/sqrt({head_size})')
print(f'   divides variance by {head_size} (because Var(aX) = a²Var(X))')
print(f'   {head_size} * ({scale_factor:.4f})² = {head_size * scale_factor**2:.4f} ≈ 1')
```


**Output:**
```
the solution: scaled attention

scale factor = 1 / sqrt(head_size) = 1 / sqrt(16) = 0.2500

wei_scaled = (q @ k.transpose(-2, -1)) * 0.2500

wei_scaled variance: 0.7791

the variance is back to ~1!

why does this work?
   unscaled variance ≈ 16
   multiplying by 0.2500 = 1/sqrt(16)
   divides variance by 16 (because Var(aX) = a²Var(X))
   16 * (0.2500)² = 1.0000 ≈ 1

```


### Why Scaling Matters for Softmax
When values are too large, softmax becomes "peaky" - almost all probability goes to one position.
When values are small, softmax is "diffuse" - probability spreads more evenly.

For learning to work well, we need.
- Gradients to flow to multiple positions
- The network to learn which positions to attend to
- Smooth, not sharp, attention distributions early in training


##### let's see what happens to softmax when values are too large


```python
# create example with small values
print('example: small values going into softmax')
print()
small_values = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])
print(f'small_values = {small_values.tolist()}')
print()
print('these values have variance close to the expected ~1 from scaling')
print(f'   variance: {small_values.var().item():.4f}')
print(f'   max absolute value: {small_values.abs().max().item():.4f}')
```


**Output:**
```
example: small values going into softmax

small_values = [0.10000000149011612, -0.20000000298023224, 0.30000001192092896, -0.20000000298023224, 0.5]

these values have variance close to the expected ~1 from scaling
   variance: 0.0950
   max absolute value: 0.5000

```


```python
# apply softmax to small values
print('applying softmax to small values')
print()
softmax_small = torch.softmax(small_values, dim=-1)
print(f'input:   {small_values.tolist()}')
print(f'softmax: {[round(x, 4) for x in softmax_small.tolist()]}')
print()
print('tracing through the softmax calculation')
import math
exp_vals = [math.exp(v.item()) for v in small_values]
exp_sum = sum(exp_vals)
print(f'   e^0.1 = {exp_vals[0]:.4f}')
print(f'   e^-0.2 = {exp_vals[1]:.4f}')
print(f'   e^0.3 = {exp_vals[2]:.4f}')
print(f'   e^-0.2 = {exp_vals[3]:.4f}')
print(f'   e^0.5 = {exp_vals[4]:.4f}')
print(f'   sum = {exp_sum:.4f}')
print()
print('softmax = e^x / sum(e^x)')
for i, (exp_v, soft_v) in enumerate(zip(exp_vals, softmax_small.tolist())):
    print(f'   position {i}: {exp_v:.4f} / {exp_sum:.4f} = {soft_v:.4f}')
```


**Output:**
```
applying softmax to small values

input:   [0.10000000149011612, -0.20000000298023224, 0.30000001192092896, -0.20000000298023224, 0.5]
softmax: [0.1925, 0.1426, 0.2351, 0.1426, 0.2872]

tracing through the softmax calculation
   e^0.1 = 1.1052
   e^-0.2 = 0.8187
   e^0.3 = 1.3499
   e^-0.2 = 0.8187
   e^0.5 = 1.6487
   sum = 5.7412

softmax = e^x / sum(e^x)
   position 0: 1.1052 / 5.7412 = 0.1925
   position 1: 0.8187 / 5.7412 = 0.1426
   position 2: 1.3499 / 5.7412 = 0.2351
   position 3: 0.8187 / 5.7412 = 0.1426
   position 4: 1.6487 / 5.7412 = 0.2872

```


```python
# analyze the small values softmax result
print('analyzing small values softmax result')
print()
print(f'softmax: {[round(x, 4) for x in softmax_small.tolist()]}')
print()
print('observations')
print(f'   min probability: {softmax_small.min().item():.4f}')
print(f'   max probability: {softmax_small.max().item():.4f}')
print(f'   ratio max/min:   {softmax_small.max().item() / softmax_small.min().item():.2f}x')
print()
print('the distribution is DIFFUSE (spread out)')
print('   - all positions get meaningful probability')
print('   - no single position dominates')
print('   - gradients can flow to all positions')
print('   - this is GOOD for learning!')
```


**Output:**
```
analyzing small values softmax result

softmax: [0.1925, 0.1426, 0.2351, 0.1426, 0.2872]

observations
   min probability: 0.1426
   max probability: 0.2872
   ratio max/min:   2.01x

the distribution is DIFFUSE (spread out)
   - all positions get meaningful probability
   - no single position dominates
   - gradients can flow to all positions
   - this is GOOD for learning!

```


```python
# create example with large values
print('example: large values going into softmax')
print()
large_values = small_values * 8
print(f'large_values = small_values * 8')
print(f'            = {large_values.tolist()}')
print()
print('this simulates what happens WITHOUT scaling')
print('if head_size = 64, values could be 8x larger')
print(f'   variance: {large_values.var().item():.4f}')
print(f'   max absolute value: {large_values.abs().max().item():.4f}')
```


**Output:**
```
example: large values going into softmax

large_values = small_values * 8
            = [0.800000011920929, -1.600000023841858, 2.4000000953674316, -1.600000023841858, 4.0]

this simulates what happens WITHOUT scaling
if head_size = 64, values could be 8x larger
   variance: 6.0800
   max absolute value: 4.0000

```


```python
# apply softmax to large values
print('applying softmax to large values')
print()
softmax_large = torch.softmax(large_values, dim=-1)
print(f'input:   {large_values.tolist()}')
print(f'softmax: {[round(x, 4) for x in softmax_large.tolist()]}')
print()
print('tracing through the softmax calculation')
exp_vals_large = [math.exp(v.item()) for v in large_values]
exp_sum_large = sum(exp_vals_large)
print(f'   e^0.8 = {exp_vals_large[0]:.4f}')
print(f'   e^-1.6 = {exp_vals_large[1]:.4f}')
print(f'   e^2.4 = {exp_vals_large[2]:.4f}')
print(f'   e^-1.6 = {exp_vals_large[3]:.4f}')
print(f'   e^4.0 = {exp_vals_large[4]:.4f}')
print(f'   sum = {exp_sum_large:.4f}')
print()
print('notice how e^4.0 = {:.2f} dominates the sum!'.format(exp_vals_large[4]))
```


**Output:**
```
applying softmax to large values

input:   [0.800000011920929, -1.600000023841858, 2.4000000953674316, -1.600000023841858, 4.0]
softmax: [0.0326, 0.003, 0.1615, 0.003, 0.8]

tracing through the softmax calculation
   e^0.8 = 2.2255
   e^-1.6 = 0.2019
   e^2.4 = 11.0232
   e^-1.6 = 0.2019
   e^4.0 = 54.5982
   sum = 68.2507

notice how e^4.0 = 54.60 dominates the sum!

```


```python
# analyze the large values softmax result
print('analyzing large values softmax result')
print()
print(f'softmax: {[round(x, 4) for x in softmax_large.tolist()]}')
print()
print('observations')
print(f'   min probability: {softmax_large.min().item():.6f}')
print(f'   max probability: {softmax_large.max().item():.4f}')
print(f'   ratio max/min:   {softmax_large.max().item() / softmax_large.min().item():.0f}x')
print()
print('the distribution is PEAKY (concentrated)')
print('   - position 4 gets almost all probability (~0.96)')
print('   - other positions get almost nothing')
print('   - gradients only flow to position 4')
print('   - this is BAD for learning!')
print()
print('this is why we scale by 1/sqrt(head_size)')
print('   - keeps attention scores in a reasonable range')
print('   - prevents softmax from becoming too peaky')
print('   - allows gradients to flow during training')
```


**Output:**
```
analyzing large values softmax result

softmax: [0.0326, 0.003, 0.1615, 0.003, 0.8]

observations
   min probability: 0.002958
   max probability: 0.8000
   ratio max/min:   270x

the distribution is PEAKY (concentrated)
   - position 4 gets almost all probability (~0.96)
   - other positions get almost nothing
   - gradients only flow to position 4
   - this is BAD for learning!

this is why we scale by 1/sqrt(head_size)
   - keeps attention scores in a reasonable range
   - prevents softmax from becoming too peaky
   - allows gradients to flow during training

```


### Step 9: Layer Normalization

#### What is Layer Normalization?
**Layer Normalization** normalizes the features of each sample independently, ensuring they have mean=0 and std=1.

#### The Formula
For each sample x with features [x₁, x₂, ..., xₙ].
1. Compute mean: μ = (x₁ + x₂ + ... + xₙ) / n
2. Compute variance: σ² = Σ(xᵢ - μ)² / n
3. Normalize: x̂ᵢ = (xᵢ - μ) / √(σ² + ε)
4. Scale and shift: yᵢ = γ * x̂ᵢ + β

Where.
- ε is a small constant for numerical stability (avoid divide by zero)
- γ (gamma) and β (beta) are learnable parameters

#### Why Do We Need Layer Normalization?
Without normalization, activations can.
- Explode (grow very large) → gradients explode
- Vanish (become very small) → gradients vanish
- Have very different scales → training instability

Layer normalization keeps activations in a stable range, making training faster and more reliable.

#### BatchNorm vs LayerNorm

| Normalization | Normalizes across | Used in |
|--------------|-------------------|---------|
| **BatchNorm** | Same feature across batch samples (column-wise) | CNNs |
| **LayerNorm** | All features within one sample (row-wise) | Transformers |

#### Why LayerNorm for Transformers?
LayerNorm is preferred in Transformers because.
1. Works with any batch size (including batch_size=1 for inference)
2. Each token's representation is normalized independently
3. No dependency on other samples in the batch


```python
class LayerNorm1d:
    """
    Layer Normalization: normalizes each sample independently.
    
    For input of shape (batch_size, features):
    - Each row (sample) is normalized to have mean=0, std=1
    - Then scaled by learnable gamma and shifted by learnable beta
    """

    def __init__(self, dim, eps=1e-5):
        """
        Initialize the LayerNorm1d layer.
        
        Args:
            dim: Number of features.
            eps: Small constant for numerical stability (avoid divide by zero).
        """
        self.eps = eps
        # learnable parameters
        self.gamma = torch.ones(dim)   # scale (initialized to 1)
        self.beta = torch.zeros(dim)   # shift (initialized to 0)

    def __call__(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, dim).
            
        Returns:
            Normalized tensor of same shape.
        """
        # step 1: compute mean of each sample (across features)
        xmean = x.mean(dim=1, keepdim=True)  # (batch, 1)
        # step 2: compute variance of each sample
        xvar = x.var(dim=1, keepdim=True)    # (batch, 1)
        # step 3: normalize to mean=0, var=1
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        # step 4: scale and shift (learnable)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        """
        Return learnable parameters.
        
        Returns:
            List containing gamma (scale) and beta (shift) tensors.
        """
        return [self.gamma, self.beta]
```


```python
# understanding the LayerNorm class step by step
print('understanding the LayerNorm1d class')
print()
print('__init__(self, dim, eps=1e-5)')
print('   dim: number of features to normalize over')
print('   eps: small constant to avoid division by zero')
print('   gamma: learnable scale parameter, initialized to 1')
print('   beta: learnable shift parameter, initialized to 0')
print()
print('__call__(self, x)')
print('   step 1: compute mean across features (dim=1)')
print('           xmean = x.mean(dim=1, keepdim=True)')
print()
print('   step 2: compute variance across features (dim=1)')
print('           xvar = x.var(dim=1, keepdim=True)')
print()
print('   step 3: normalize to mean=0, variance=1')
print('           xhat = (x - xmean) / sqrt(xvar + eps)')
print()
print('   step 4: scale and shift (learnable)')
print('           out = gamma * xhat + beta')
print()
print('parameters()')
print('   returns [gamma, beta] for optimization')
```


**Output:**
```
understanding the LayerNorm1d class

__init__(self, dim, eps=1e-5)
   dim: number of features to normalize over
   eps: small constant to avoid division by zero
   gamma: learnable scale parameter, initialized to 1
   beta: learnable shift parameter, initialized to 0

__call__(self, x)
   step 1: compute mean across features (dim=1)
           xmean = x.mean(dim=1, keepdim=True)

   step 2: compute variance across features (dim=1)
           xvar = x.var(dim=1, keepdim=True)

   step 3: normalize to mean=0, variance=1
           xhat = (x - xmean) / sqrt(xvar + eps)

   step 4: scale and shift (learnable)
           out = gamma * xhat + beta

parameters()
   returns [gamma, beta] for optimization

```


```python
# test layer normalization
torch.manual_seed(42)

# create the layer
n_features = 100
module = LayerNorm1d(n_features)
print(f'created LayerNorm1d with {n_features} features')
print(f'   gamma shape: {module.gamma.shape}')
print(f'   beta shape:  {module.beta.shape}')
print()

# create test data
n_samples = 32
x = torch.randn(n_samples, n_features)
print(f'created test data with shape: {x.shape}')
print(f'   {n_samples} samples, {n_features} features each')
```


**Output:**
```
created LayerNorm1d with 100 features
   gamma shape: torch.Size([100])
   beta shape:  torch.Size([100])

created test data with shape: torch.Size([32, 100])
   32 samples, 100 features each

```


```python
# examine the data BEFORE normalization
print('examining data BEFORE normalization')
print()
print('sample 0')
print(f'   mean: {x[0].mean().item():.4f}')
print(f'   std:  {x[0].std().item():.4f}')
print(f'   min:  {x[0].min().item():.4f}')
print(f'   max:  {x[0].max().item():.4f}')
print()
print('sample 1')
print(f'   mean: {x[1].mean().item():.4f}')
print(f'   std:  {x[1].std().item():.4f}')
print(f'   min:  {x[1].min().item():.4f}')
print(f'   max:  {x[1].max().item():.4f}')
print()
print('sample 2')
print(f'   mean: {x[2].mean().item():.4f}')
print(f'   std:  {x[2].std().item():.4f}')
print(f'   min:  {x[2].min().item():.4f}')
print(f'   max:  {x[2].max().item():.4f}')
print()
print('note: means are not 0, stds are not exactly 1')
print('each sample has slightly different statistics')
```


**Output:**
```
examining data BEFORE normalization

sample 0
   mean: 0.0320
   std:  1.0406
   min:  -2.5095
   max:  2.2181

sample 1
   mean: 0.0811
   std:  0.9597
   min:  -2.4801
   max:  2.0265

sample 2
   mean: 0.0571
   std:  0.9930
   min:  -2.5850
   max:  2.1120

note: means are not 0, stds are not exactly 1
each sample has slightly different statistics

```


```python
# apply layer normalization
x_normed = module(x)
print('applied layer normalization')
print(f'   x_normed shape: {x_normed.shape}')
```


**Output:**
```
applied layer normalization
   x_normed shape: torch.Size([32, 100])

```


```python
# examine the data AFTER normalization
print('examining data AFTER normalization')
print()
print('sample 0')
print(f'   mean: {x_normed[0].mean().item():.6f}')
print(f'   std:  {x_normed[0].std().item():.4f}')
print(f'   min:  {x_normed[0].min().item():.4f}')
print(f'   max:  {x_normed[0].max().item():.4f}')
print()
print('sample 1')
print(f'   mean: {x_normed[1].mean().item():.6f}')
print(f'   std:  {x_normed[1].std().item():.4f}')
print(f'   min:  {x_normed[1].min().item():.4f}')
print(f'   max:  {x_normed[1].max().item():.4f}')
print()
print('sample 2')
print(f'   mean: {x_normed[2].mean().item():.6f}')
print(f'   std:  {x_normed[2].std().item():.4f}')
print(f'   min:  {x_normed[2].min().item():.4f}')
print(f'   max:  {x_normed[2].max().item():.4f}')
print()
print('now all samples have mean ≈ 0 and std ≈ 1!')
print('layer normalization standardizes each sample independently')
```


**Output:**
```
examining data AFTER normalization

sample 0
   mean: -0.000000
   std:  1.0000
   min:  -2.4425
   max:  2.1008

sample 1
   mean: 0.000000
   std:  1.0000
   min:  -2.6687
   max:  2.0271

sample 2
   mean: 0.000000
   std:  1.0000
   min:  -2.6605
   max:  2.0694

now all samples have mean ≈ 0 and std ≈ 1!
layer normalization standardizes each sample independently

```


```python
# verify normalization for all samples
print('verifying normalization for all samples')
print()
means = x_normed.mean(dim=1)
stds = x_normed.std(dim=1)
print(f'mean of all sample means: {means.mean().item():.8f} (should be ~0)')
print(f'mean of all sample stds:  {stds.mean().item():.4f} (should be ~1)')
print()
print('all samples are now normalized!')
```


**Output:**
```
verifying normalization for all samples

mean of all sample means: 0.00000000 (should be ~0)
mean of all sample stds:  1.0000 (should be ~1)

all samples are now normalized!

```


### Understanding the Difference: BatchNorm vs LayerNorm
Let's visualize what each normalization normalizes over.

For data of shape (batch_size, features) = (32, 100).
- **BatchNorm**: Normalizes each FEATURE across all samples
  - For feature 0: compute mean and std across all 32 samples
  - Result: each feature has mean=0, std=1 across the batch
  
- **LayerNorm**: Normalizes each SAMPLE across all features
  - For sample 0: compute mean and std across all 100 features
  - Result: each sample has mean=0, std=1 across its features


```python
# demonstrate the difference between BatchNorm and LayerNorm
print('demonstrating the difference between BatchNorm and LayerNorm')
print()
print('our data has shape (32, 100) = (batch_size, features)')
print()
print('LAYERNORM normalizes across FEATURES (row-wise)')
print('checking ONE SAMPLE across all features')
print(f'   sample 0 - mean: {x_normed[0,:].mean().item():.6f}, std: {x_normed[0,:].std().item():.4f}')
print(f'   sample 1 - mean: {x_normed[1,:].mean().item():.6f}, std: {x_normed[1,:].std().item():.4f}')
print(f'   sample 2 - mean: {x_normed[2,:].mean().item():.6f}, std: {x_normed[2,:].std().item():.4f}')
print('   → all samples are normalized (mean≈0, std≈1)')
print()
print('BATCHNORM would normalize across SAMPLES (column-wise)')
print('checking ONE FEATURE across all samples')
print(f'   feature 0 - mean: {x_normed[:,0].mean().item():.4f}, std: {x_normed[:,0].std().item():.4f}')
print(f'   feature 1 - mean: {x_normed[:,1].mean().item():.4f}, std: {x_normed[:,1].std().item():.4f}')
print(f'   feature 2 - mean: {x_normed[:,2].mean().item():.4f}, std: {x_normed[:,2].std().item():.4f}')
print('   → features are NOT normalized (this is what BatchNorm would normalize)')
print()
print('LayerNorm normalizes each sample independently')
print('BatchNorm would normalize each feature across the batch')
```


**Output:**
```
demonstrating the difference between BatchNorm and LayerNorm

our data has shape (32, 100) = (batch_size, features)

LAYERNORM normalizes across FEATURES (row-wise)
checking ONE SAMPLE across all features
   sample 0 - mean: -0.000000, std: 1.0000
   sample 1 - mean: 0.000000, std: 1.0000
   sample 2 - mean: 0.000000, std: 1.0000
   → all samples are normalized (mean≈0, std≈1)

BATCHNORM would normalize across SAMPLES (column-wise)
checking ONE FEATURE across all samples
   feature 0 - mean: -0.1428, std: 1.0629
   feature 1 - mean: 0.0941, std: 1.1109
   feature 2 - mean: 0.2947, std: 1.1421
   → features are NOT normalized (this is what BatchNorm would normalize)

LayerNorm normalizes each sample independently
BatchNorm would normalize each feature across the batch

```


```python
# why LayerNorm is preferred in Transformers
print('why LayerNorm is preferred in Transformers')
print()
print('1. works with any batch size')
print('   - BatchNorm needs multiple samples to compute batch statistics')
print('   - LayerNorm only needs one sample (normalizes across features)')
print('   - important for inference with batch_size=1')
print()
print('2. each token is normalized independently')
print('   - in Transformers, each position is like a "sample"')
print('   - we want each token\'s representation to be well-conditioned')
print('   - no dependency on other samples in the batch')
print()
print('3. no running statistics needed')
print('   - BatchNorm maintains running mean/variance during training')
print('   - LayerNorm computes fresh statistics each forward pass')
print('   - simpler implementation and behavior')
```


**Output:**
```
why LayerNorm is preferred in Transformers

1. works with any batch size
   - BatchNorm needs multiple samples to compute batch statistics
   - LayerNorm only needs one sample (normalizes across features)
   - important for inference with batch_size=1

2. each token is normalized independently
   - in Transformers, each position is like a "sample"
   - we want each token's representation to be well-conditioned
   - no dependency on other samples in the batch

3. no running statistics needed
   - BatchNorm maintains running mean/variance during training
   - LayerNorm computes fresh statistics each forward pass
   - simpler implementation and behavior

```


```python
# final summary of Part 6
print('SUMMARY: Full Self-Attention with Q, K, V')
print('=' * 60)
print()
print('SELF-ATTENTION MECHANISM')
print('   step 1: project input to queries, keys, values')
print('           Q = query(x)  # "what am I looking for?"')
print('           K = key(x)    # "what do I contain?"')
print('           V = value(x)  # "what will I give?"')
print()
print('   step 2: compute attention scores')
print('           wei = Q @ K^T')
print('           wei[i,j] = how much position i attends to position j')
print()
print('   step 3: scale by 1/sqrt(head_size)')
print('           wei = wei * (head_size ** -0.5)')
print('           prevents variance explosion')
print()
print('   step 4: mask future positions')
print('           wei = wei.masked_fill(tril == 0, -inf)')
print('           prevents looking at future tokens')
print()
print('   step 5: apply softmax')
print('           wei = softmax(wei)')
print('           converts to probabilities that sum to 1')
print()
print('   step 6: weighted sum of values')
print('           out = wei @ V')
print('           each position gets weighted average of values')
print()
print('LAYER NORMALIZATION')
print('   normalizes each sample to mean=0, std=1')
print('   stabilizes training by keeping activations in good range')
print('   applied before attention and feedforward layers')
print()
print('COMPLETE ATTENTION FORMULA')
print('   Attention(Q, K, V) = softmax(mask(Q @ K^T / sqrt(d_k))) @ V')
```


**Output:**
```
SUMMARY: Full Self-Attention with Q, K, V
============================================================

SELF-ATTENTION MECHANISM
   step 1: project input to queries, keys, values
           Q = query(x)  # "what am I looking for?"
           K = key(x)    # "what do I contain?"
           V = value(x)  # "what will I give?"

   step 2: compute attention scores
           wei = Q @ K^T
           wei[i,j] = how much position i attends to position j

   step 3: scale by 1/sqrt(head_size)
           wei = wei * (head_size ** -0.5)
           prevents variance explosion

   step 4: mask future positions
           wei = wei.masked_fill(tril == 0, -inf)
           prevents looking at future tokens

   step 5: apply softmax
           wei = softmax(wei)
           converts to probabilities that sum to 1

   step 6: weighted sum of values
           out = wei @ V
           each position gets weighted average of values

LAYER NORMALIZATION
   normalizes each sample to mean=0, std=1
   stabilizes training by keeping activations in good range
   applied before attention and feedforward layers

COMPLETE ATTENTION FORMULA
   Attention(Q, K, V) = softmax(mask(Q @ K^T / sqrt(d_k))) @ V

```


## MIT License

