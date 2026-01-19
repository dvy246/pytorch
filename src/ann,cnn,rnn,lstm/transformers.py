import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import math

class InputEmbeddings(nn.Module):
    """
    A class to handle input embeddings for a transformer model.

    This module converts token indices into dense vectors of fixed size using an embedding layer.
    The embeddings are scaled by the square root of the model dimension to stabilize gradients.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimension of the model.
        embedding (nn.Embedding): The embedding layer that maps token indices to vectors.
    """
    def __init__(self, vocab_size: int, d_model: int):
        """
        Initializes the InputEmbeddings module.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimension of the model.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the InputEmbeddings module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing token indices.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model) containing the embedded vectors,
                         scaled by the square root of the model dimension.
        """
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self,seq_length,d_model:int,dropout:float=0.1):
        super().__init__()
        self.seq_length=seq_length
        self.d_model=d_model
        self.dropout=nn.Dropout(dropout)


        # a matrix of (seq_lenth,d_model) which is of the same size of a embeddings and of max length of a sentence 
        pe=torch.zeros(self.seq_length,self.d_model)

        #positional vector of size (seq_lenght,1)
        postion=torch.arange(0,self.seq_length,dtype=torch.float32).unsqueeze(1)
        #divisor of size (1,d_model)
        div_term=torch.exp(torch.arange(0.,self.d_model,2).float()*(-math.log(10000.0)/self.d_model))
        #apply the sin to even positions
        pe[:,0::2]=torch.sin(postion*div_term)
        pe[:,1::2]=torch.cos(postion*div_term)

        #make the pe a batch by adding one more dimension
        pe=pe.unsqueeze(0) #(1,seq_length,d_model)

        #to save a tensor which is not learned 
        self.register_buffer('pe',pe)

    def forward(self,x):
          x=x+(self.pe[:,:x.shape(1),:]).requires_grad(False)
          return self.dropout(x)
        