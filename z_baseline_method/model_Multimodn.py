import torch
import torch.nn as nn
import einops
from typing import *
# from healnet.baselines.multimodn.encoders import MultiModEncoder
# from healnet.baselines.multimodn.decoders import MultiModDecoder
# from healnet.baselines.multimodn.utils import TrainableInitState

from torch import Tensor
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional
from torchvision.models import resnet18, ResNet18_Weights
from torch import sigmoid

class InitState(nn.Module, ABC):
    """Trainable initial state"""

    def __init__(self, state_size: int):
        super().__init__()
        self.state_size = state_size

    @abstractmethod
    def forward(self, batch_size) -> Tensor:
        pass


class TrainableInitState(InitState):
    """Trainable initial state"""

    def __init__(self, state_size: int, device: Optional[torch.device] = None):
        super().__init__(state_size)
        self.device = device
        self.state_value = nn.Parameter(
            torch.randn((1, state_size), requires_grad=True, device=self.device)
        )

    def forward(self, batch_size) -> Tensor:
        init_tensor = torch.tile(self.state_value, [batch_size, 1])

        return init_tensor


class MultiModDecoder(nn.Module, ABC):
    """Abstract decoder for MultiModN"""

    def __init__(self, state_size: int):
        super(MultiModDecoder, self).__init__()
        self.state_size = state_size

    @abstractmethod
    def forward(self, state: Tensor) -> Tensor:
        pass


class ClassDecoder(MultiModDecoder):
    """Classifier for MultiModN"""

    def __init__(self, state_size: int, n_classes: int, activation: Callable,
                 device: Optional[torch.device] = None):
        super().__init__(state_size)
        self.n_classes = n_classes
        self.fc = nn.Linear(state_size, n_classes, device=device)
        self.activation = activation

    def forward(self, state: Tensor) -> Tensor:
        return self.activation(self.fc(state))
    
class MLPDecoder(MultiModDecoder):
    """Multi-layer perceptron decoder"""
    def __init__(
            self,
            state_size: int,            
            hidden_layers: Tuple[int],
            n_classes: int = 2,
            output_activation: Callable = sigmoid,
            hidden_activation: Callable = F.relu,
            device: Optional[torch.device] = None,
    ):
        super().__init__(state_size)
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.n_classes = n_classes
        dim_layers = [self.state_size] + list(hidden_layers) + [n_classes, ]
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dim_layers, dim_layers[1:])):  
            self.layers.append(nn.Linear(in_dim, out_dim, device=device))

    def forward(self, latent: Tensor) -> Tensor:

        # # expand to num_classes
        # latent = einops.repeat(latent, "d -> b d", b = self.n_classes)

        for layer in self.layers[0:-1]:
            latent = self.hidden_activation((layer(latent)))
        output = self.output_activation(self.layers[-1](latent))
        return output


class LogisticDecoder(ClassDecoder):
    """Logistic decoder for MultiModN"""

    def __init__(self, state_size: int, device: Optional[torch.device] = None):
        super().__init__(state_size, 2, sigmoid, device)


class MultiModEncoder(nn.Module, ABC):
    """Abstract encoder for MultiModN"""

    def __init__(self, state_size: int):
        super(MultiModEncoder, self).__init__()
        self.state_size = state_size

    @abstractmethod
    def forward(self, state: Tensor, x: Tensor) -> Tensor:
        pass




class MLPEncoder(MultiModEncoder):
    """Multi-layer perceptron encoder"""
    def __init__(
            self,
            state_size: int,
            n_features: int,
            hidden_layers: Tuple[int],
            activation: Callable = F.relu,
            device: Optional[torch.device] = None,
    ):
        super().__init__(state_size)

        self.activation = activation

        dim_layers = [n_features] + list(hidden_layers) + [self.state_size, ]

        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dim_layers, dim_layers[1:])):
            # The state is concatenated to the input of the last layer
            if i == len(dim_layers) - 2:
                self.layers.append(
                    nn.Linear(in_dim + self.state_size, out_dim, device=device))
            else:
                self.layers.append(nn.Linear(in_dim, out_dim, device=device))

    def forward(self, state: Tensor, x: Tensor) -> Tensor:
        # b, *_ = x.shape
        # state = einops.repeat(state, "d -> b d", b=b)

        for layer in self.layers[0:-1]:
            x = self.activation(layer(x))

        output = self.layers[-1](torch.cat([x, state], dim=1))

        # reduce state over batch
        # output = nn.Parameter(einops.reduce(output, "b d -> d", "mean"))

        return nn.Parameter(output)


class PatchEncoder(MultiModEncoder):
    """RNN encoder adjusted for patched images"""

    def __init__(
        self,
        state_size: int,
        n_features: int,
        hidden_layers: Tuple[int],
        activation: Callable = F.relu,
    ):
        super().__init__(state_size)

        self.activation = activation

        dim_layers = [n_features] + list(hidden_layers) + [self.state_size,]

        self.layers = nn.ModuleList()
        for i, (inDim, outDim) in enumerate(zip(dim_layers, dim_layers[1:])):
            # The state is concatenated to the input of the last layer
            if i == len(dim_layers)-2:
                self.layers.append(nn.RNN(inDim + self.state_size, outDim, batch_first=True))
            else:
                self.layers.append(nn.RNN(inDim, outDim, batch_first=True))

    def forward(self, state: Tensor, x: Tensor) -> Tensor:
        # expand state
        # b, *_ = x.shape
        # state = einops.repeat(state, "d -> b d", b=b)

        for layer in self.layers[:-1]:
            out, h_n = layer(x)
            x = self.activation(out)

        # need to average over patches
        output, h_n = self.layers[-1](torch.cat([einops.reduce(tensor=x, pattern="b c d -> b d", reduction="sum"), state], dim=1))

        # reduce state over batch
        # output = nn.Parameter(einops.reduce(output, "b d -> d", "mean"))

        return nn.Parameter(output)



class ResNet(nn.Module):
    def __init__(self, *, state_size=0, freeze=False, pretrained_path=None, pretrained=True):
        super().__init__()

        if pretrained_path is not None and pretrained:
            raise ValueError(
                "Loading a pretrained ResNet should either be from torch.vision (pretrained=True) "
                "or from a checkpoint (pretrained_path) but not both."
            )

        # if pretrained, loads ResNet18 pretrained on ImageNet
        # self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.state_size = state_size

        self.fc = nn.Linear(512 + self.state_size, self.state_size)

        # load pre-trained ResNet from path
        if pretrained_path:
            model_dict = self.resnet.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {
                k: v for k, v in torch.load(pretrained_path).items() if k in model_dict
            }
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.resnet.load_state_dict(model_dict)

        # remove final classification layer
        self.resnet.fc = nn.Identity()

        if freeze:
            for p in self.resnet.parameters():
                p.requires_grad = False

    def forward(self, state, x):
        # expand state

        representations = self.resnet(x)
        output = self.fc(torch.cat([representations, state], dim=1))

        return nn.Parameter(output)


class MultiModNModule(nn.Module):
    def __init__(self,
                 state_size: int,
                 encoders: List[MultiModEncoder], # needs to be in right order of modalities in x
                 decoders: List[MultiModDecoder], # just 1 in our case
                 err_penalty: float = 1.0, # from main pipeline
                 state_change_penalty: float = 0.0,
                 ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        # self.state = TrainableInitState(state_size=state_size, device=self.device)
        self.state = nn.Parameter(torch.randn(state_size), requires_grad=True)
        self.encoders = encoders
        self.decoders = decoders
        self.err_penalty = err_penalty
        self.state_change_penalty = state_change_penalty


        self.model = nn.ModuleList(encoders)
        self.decoder = nn.ModuleList(decoders)

    def forward(self, 
                #x,y, 
                x: List[torch.Tensor],
                target: torch.Tensor) -> torch.Tensor: #x: List[torch.Tensor],
        
        # x = [x,y]
        assert len(x) == len(self.encoders), "Number of inputs must match number of encoders"
        
        b, *_ = x[0].shape # get batch dims

        # each sample in batch gets assigned state
        # only expand once
        # if len(self.state.shape) == 1:
        self.state = nn.Parameter(einops.repeat(self.state, "d -> b d", b=b))


        running_loss = 0
        for encoder, mod in zip(self.encoders, x):
            old_state = self.state.clone()
            self.state = encoder(state=self.state, x=mod) # (l_d)

        # iterate through decoders as it's multitask
            for decoder in self.decoders:
                predict = decoder(self.state)
                loss = self.calc_loss(predict, target, old_state, self.state)
            running_loss += loss

        running_loss /= len(self.encoders)
        # reduce state over batch dim
        self.state = nn.Parameter(self.state.mean(dim=0))
        # return expected loss over batches and encoders and predictions after the last state (encoder)
        Y_hat = torch.topk(predict, 1, dim=1)[1]
        Y_prob = F.softmax(predict, dim=1)
        hazards = torch.sigmoid(predict)
        S = torch.cumprod(1 - hazards, dim=1)
        risk_scores = -torch.sum(S, dim=1)
        return running_loss, hazards, S, Y_hat
        # return running_loss, pred



    def calc_loss(self, pred: torch.Tensor, actual: torch.Tensor, s_old: torch.Tensor, s_new: torch.Tensor):
        b, *_ = pred.shape
        err_loss = nn.CrossEntropyLoss()(pred, actual.float())
        state_change_loss = torch.mean((s_new - s_old) ** 2)

        # mean over mini-batch
        loss = torch.mean((err_loss * self.err_penalty + state_change_loss * self.state_change_penalty), dim=0)

        return loss


