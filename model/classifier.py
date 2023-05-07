from torch import nn, optim
from model.base_model import base_model
import torch


class Softmax_Layer(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, input_size, num_class):
        """
        Args:
            num_class: number of classes
        """
        super(Softmax_Layer, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.fc = nn.Linear(self.input_size, self.num_class, bias=False)

    def forward(self, input):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        logits = self.fc(input)
        return logits


class Proto_Softmax_Layer(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, config):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super(Proto_Softmax_Layer, self).__init__()
        self.config = config

    def set_prototypes(self, protos):
        self.prototypes = protos.to(self.config.device)

    def forward(self, rep):

        dis_mem = self.__distance__(rep, self.prototypes)
        return dis_mem

    def __distance__(self, rep, rel):
        '''
        rep_ = rep.view(rep.shape[0], 1, rep.shape[-1])
        rel_ = rel.view(1, -1, rel.shape[-1])
        dis = (rep_ * rel_).sum(-1)
        return dis
        '''
        rep_norm = rep / rep.norm(dim=1)[:, None]
        rel_norm = rel / rel.norm(dim=1)[:, None]

        res = torch.mm(rep_norm, rel_norm.transpose(0, 1))
        return res