import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)

    return x.flatten(-2)

def apply_rotation(x, sin, cos):
    return (x * cos + rotate_every_two(x) * sin)

class RetNet_Pos(nn.Module):
    """
    RetNets Position Encoding
    Based on the RoPE encoding 

    X : Matrix/Vector to apply embeddings on
    """

    def __init__(self, hidden_dim):
  
        super(RetNet_Pos, self).__init__()
        angle = (1 / (10000 ** torch.linspace(0, 1, hidden_dim // 2)))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    def forward(self, X, slen=1, recurrence=False, conjugate=False):

        if recurrence == False:

          index = torch.arange(slen).to(self.angle)
          angles = index[:,None].to(self.angle) @ self.angle[None , :] 
          sin = torch.sin(angles)
          cos = torch.cos(angles)
        
        else:

          angles = self.angle * (slen - 1)
          sin = torch.sin(angles)
          cos = torch.cos(angles)

        if conjugate == False:

          return apply_rotation(X, sin, cos)

        else:

          return apply_rotation(X, -1 * sin , cos)


class Retention(nn.Module):
    """
    compute retention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, hidden_dim, head_size=None, gamma = 0.1, double_v=True,
                 chunk_size=None):
        super(Retention, self).__init__()
        self.gamma  = gamma
        self.hidden_dim = hidden_dim

        if head_size is None:
          head_size = hidden_dim
        self.head_size = head_size

        if double_v is True:
          self.v_dim  = 2 * head_size
        else:
          self.v_dim = head_size
        
        self.W_Q = nn.Parameter(torch.randn(hidden_dim, head_size) / hidden_dim)
        self.W_K = nn.Parameter(torch.randn(hidden_dim, head_size) / hidden_dim)
        self.W_V = nn.Parameter(torch.randn(hidden_dim, self.v_dim) / hidden_dim)

        self.XPos = RetNet_Pos(hidden_dim)
        self.chunk_size = chunk_size
    def forward(self, X, n=1, past_kv=None , mask=None):

      q = X @ self.W_Q
      k = X @ self.W_K
      v = X @ self.W_V
      
      if self.training:

          q = self.XPos(q, X.size()[-2])
          k = self.XPos(k, X.size()[-2], conjugate=True)

          # size : [batch_size, head, length, d_tensor]
          batch_size, head, length, d_tensor = k.size()

          # 1. dot product Query with Key^T to compute similarity
          k_t = k.transpose(-2, -1)  # transpose
          score = (q @ k_t) / math.sqrt(d_tensor) # dot product

          D = self.get_D(score.size())
          score  =  score * D

          for n in range(0,score.size()[-1]):
            score[:,:,n,:] /=  max(torch.abs(torch.sum(D[n,:])) , 1)

          # 2. apply masking (the gamma mask)
          if mask is not None:
              score = score.masked_fill(mask == 0, -10000)

          # 4. multiply with Value
          vt = score @ v

          if self.chunk_size:

              assert past_kv is not None
              assert length == self.chunk_size
              
              cross_retention =  (q @ past_kv)
              cross_retention = cross_retention * self.get_eps(cross_retention.size())
              retention  = v + cross_retention

              v = v * self.get_mew(v.size())
              current_kv = k.transpose(-2,-1) @ v + past_kv * self.gamma ** self.chunk_size

              return current_kv , retention

          return vt, score

      else:
          
          assert past_kv is not None
          
          q = self.XPos(q, n, True)
          k = self.XPos(k, n, True, True)
          current_kv = self.gamma * past_kv + k.unsqueeze(-1) * v.unsqueeze(-2)
          output = (q * current_kv)

          return output, current_kv
        
    def get_D(self, size):
      D = torch.ones(size[-2],size[-1])
      for n in range(0,size[-2]):
          for m in range(0,size[-1]):
              D[n, m] = (self.gamma ** (n - m) if n >= m else 0)
          D[n,:] /=  math.sqrt(torch.sum(D[n,:]))
      return D

    def get_eps(self,size):

      eps = torch.ones(size[-2],size[-1])
      for n in range(0,size[-2]):
        eps[n,:] = self.gamma ** (n + 1)
       
      return eps

    def get_mew(self,size):

      mew = torch.ones(size[-2] , size[-1])
      for n in range(0,size[-2]):
        mew[n,:] = self.gamma ** (self.chunk_size - n - 1)