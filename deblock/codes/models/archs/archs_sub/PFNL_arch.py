''' network architecture for PFNL in PyTorch '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import models.archs.context_block as context_block

class NonLocalBlock(nn.Module):
    