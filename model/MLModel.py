# Several basic machine learning models
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import copy
from torchvision import models
import timm

class LogisticRegression(nn.Module):
    """A simple implementation of Logistic regression model"""

    def __init__(self, num_feature, output_size):
        super(LogisticRegression, self).__init__()

        self.num_feature = num_feature
        self.output_size = output_size
        self.linear = nn.Linear(self.num_feature, self.output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.linear(x)


class MLP(nn.Module):
    """A simple implementation of Deep Neural Network model"""

    def __init__(self, num_feature, output_size):
        super(MLP, self).__init__()
        self.hidden = 200
        self.model = nn.Sequential(
            nn.Linear(num_feature, self.hidden),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.hidden, output_size))

    def forward(self, x):
        return self.model(x)


class MlpModel(nn.Module):
    """
    2-hidden-layer fully connected model, 2 hidden layers with 200 units and a
    BN layer. Categorical Cross Entropy loss.
    """
    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        """
        Returns a new MNISTModelBN.
        """
        super(MlpModel, self).__init__()
        self.in_features = in_features
        self.fc0 = torch.nn.Linear(in_features, hidden_dim)
        self.relu0 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_dim, 200)
        self.relu1 = torch.nn.ReLU()
        self.out = torch.nn.Linear(200, num_classes)
        self.bn0 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn_layers = [self.bn0]

    def forward(self, x):
        """
        Returns outputs of model given data x.

        Args:
            - x: (torch.tensor) must be on same device as model

        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        x = x.reshape(-1, self.in_features)
        a = self.bn0(self.relu0(self.fc0(x)))
        b = self.relu1(self.fc1(a))

        return self.out(b)


class MnistCNN(nn.Module):
    """from fy"""
    def __init__(self, data_in, data_out):
        super(MnistCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/mlp.py
class FedAvgMLP(nn.Module):
    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/cnn.py
class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features,
                               32,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.conv2 = nn.Conv2d(32,
                               64,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.fc1 = nn.Linear(dim, 512)
        self.fc = nn.Linear(512, num_classes)

        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc(x)
        return x


"""from fy"""
class CifarCNN(nn.Module):
    def __init__(self, data_in, data_out):
        super(CifarCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = x.view(-1, 64 * 4 * 4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CifarCNN_MTFL(nn.Module):
    """
    cifar10 model of MTFL
    """

    def __init__(self, data_in, data_out):
        super(CifarCNN_MTFL, self).__init__()

        self.conv0 = torch.nn.Conv2d(3, 32, 3, 1)
        self.relu0 = torch.nn.ReLU()
        self.pool0 = torch.nn.MaxPool2d(2, 2)

        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.flat = torch.nn.Flatten()
        self.fc0 = torch.nn.Linear(2304, 512)
        self.relu2 = torch.nn.ReLU()

        self.out = torch.nn.Linear(512, 10)

        self.bn0 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm2d(64)

        # self.bn_layers = [self.bn0, self.bn1]

    def forward(self, x):
        """
        Returns outputs of model given data x.
        Args:
            - x: (torch.tensor) must be on same device as model
        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        a = self.bn0(self.pool0(self.relu0(self.conv0(x))))
        b = self.bn1(self.pool1(self.relu1(self.conv1(a))))
        c = self.relu2(self.fc0(self.flat(b)))

        return self.out(c)


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class BasicCNN(nn.Module):
    def __init__(self, data_in, data_out):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.apply(weight_init)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.fc(x)
        return x

"""Cluster FL"""
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        return x


"""FedFomo"""
class BaseConvNet(nn.Module):
    def __init__(self, in_features=1, num_classes=10, ):
        super(BaseConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""
Communication-Efficient Learning of Deep Networks from Decentralized Data
https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/models.py
"""
class CNNMnist(nn.Module):
    def __init__(self, data_in, data_out):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(data_in, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, data_out)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



"""
Communication-Efficient Learning of Deep Networks from Decentralized Data
https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/models.py
"""
class CNNFashion_Mnist(nn.Module):
    def __init__(self, data_in, data_out):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


"""
Communication-Efficient Learning of Deep Networks from Decentralized Data
https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/models.py
"""
class CNNCifar(nn.Module):
    def __init__(self, data_in, data_out):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, data_out)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# TPDS MTFL model
class CIFAR10Model(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CIFAR10Model, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 32, 3, 1)
        self.relu0 = torch.nn.ReLU()
        self.pool0 = torch.nn.MaxPool2d(2, 2)

        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.flat = torch.nn.Flatten()
        self.fc0 = torch.nn.Linear(2304, 512)
        self.relu2 = torch.nn.ReLU()

        self.out = torch.nn.Linear(512, num_classes)

        self.drop = torch.nn.Dropout(p=0.5)

        self.bn0 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm2d(64)

        self.head = [self.out]
        self.body = [self.conv0,self.conv1,self.bn0, self.bn1,self.fc0]


        # self.bn_layers = [self.bn0, self.bn1]
        self.classifier_layer = [self.fc0, self.out]

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2


    def forward(self, x):
        a = self.bn0(self.pool0(self.relu0(self.conv0(x))))
        b = self.bn1(self.pool1(self.relu1(self.conv1(a))))
        c = self.relu2(self.drop(self.fc0(self.flat(b))))
        return self.out(c)

# TPDS MTFL model
class CIFAR100Model(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CIFAR100Model, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 32, 3, 1)
        self.relu0 = torch.nn.ReLU()
        self.pool0 = torch.nn.MaxPool2d(2, 2)

        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.flat = torch.nn.Flatten()
        self.fc0 = torch.nn.Linear(2304, 512)
        self.relu2 = torch.nn.ReLU()

        self.out = torch.nn.Linear(512, 100)

        self.drop = torch.nn.Dropout(p=0.5)

        self.bn0 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm2d(64)

        # self.bn_layers = [self.bn0, self.bn1]
        self.classifier_layer = [self.fc0, self.out]
        self.head = [self.out]
        self.body = [self.conv0,self.conv1,self.bn0, self.bn1,self.fc0]

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def forward(self, x):
        a = self.bn0(self.pool0(self.relu0(self.conv0(x))))
        b = self.bn1(self.pool1(self.relu1(self.conv1(a))))
        c = self.relu2(self.drop(self.fc0(self.flat(b))))
        return self.out(c)



# from TPDS
class FashionMNISTModel(nn.Module):
    def __init__(self, num_classes):
        """
        Returns a new FashionMNISTModel.

        Args:
            - device: (torch.device) to place model on
        """
        super(FashionMNISTModel, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 32, 7, padding=3)
        self.act = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.bn0 = torch.nn.BatchNorm2d(32)
        self.conv1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.out = torch.nn.Linear(64 * 7 * 7, num_classes)
        self.bn_layers = [self.bn0, self.bn1]
        self.head = [self.out]
        self.body = [self.conv0,self.bn0,self.conv1,self.bn1]

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def forward(self, x):
        """
        Returns outputs of model given data x.

        Args:
            - x: (torch.tensor) must be on same device as model

        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        x = x.reshape(-1, 1, 28, 28)
        x = self.bn0(self.pool(self.act(self.conv0(x))))
        x = self.bn1(self.pool(self.act(self.conv1(x))))
        x = x.flatten(1)
        return self.out(x)


class FemnistCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """
    def __init__(self, num_classes):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.relu = torch.nn.ReLU()
        self.fc1 = nn.Linear(64 * 4 * 4, 2048)
        self.output = nn.Linear(2048, num_classes)
        self.classifier_layer = [self.fc1, self.output]
        self.head = [self.output]
        self.body = [self.conv1,self.conv2,self.fc1]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

class Reswithoutcon(nn.Module):
    def __init__(self, option='resnet50', pret=False, with_con=True, num_classes=10):
        super(Reswithoutcon, self).__init__()
        self.dim = 2048
        self.with_con = with_con
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret,num_classes=num_classes,zero_init_residual=True)
            self.dim = 512
        if option == 'resnet34':
            model_ft = models.resnet34(pretrained=pret,num_classes=num_classes,zero_init_residual=True)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret,num_classes=num_classes,zero_init_residual=True)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret,num_classes=num_classes,zero_init_residual=True)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret,num_classes=num_classes,zero_init_residual=True)
        
        mod = list(model_ft.children())
        if with_con:
            temp = mod.pop(0)
            self.features = model_ft
            self.body = temp
            self.head = mod
        else:
            mod = list(model_ft.children())
            mod.pop(0)
            self.class_fit = nn.Sequential(*mod)
            
    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def forward(self, x):
        # x = self.features(x)
        if self.with_con:
            x = self.features(x)
            return x
        else:
            x = self.class_fit(x)
            return x


class MobilenetV2(nn.Module):
    def __init__(self, option='MobilenetV2', pret=False, with_con=True,num_classes=10):
        super(MobilenetV2, self).__init__()
        self.dim = 2048
        self.with_con = with_con
        model_ft = models.mobilenet_v2(pretrained=pret,num_classes=num_classes)
        mod = list(model_ft.children())
        if with_con:
            temp = mod.pop(0)
            self.features = model_ft
            self.body = temp
            self.head = mod
        else:
            mod = list(model_ft.children())
            mod.pop(0)
            self.class_fit = nn.Sequential(*mod)
            
    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                for temp in bn:
                    if hasattr(temp, 'weight'):
                        vals.append(copy.deepcopy(temp.weight))
                    if hasattr(temp, 'bias'):
                        vals.append(copy.deepcopy(temp.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                for temp in bn:
                    if hasattr(temp, 'weight'):
                        vals.append(copy.deepcopy(temp.weight))
                    if hasattr(temp, 'bias'):
                        vals.append(copy.deepcopy(temp.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                for temp in bn:
                    if hasattr(temp, 'weight'):
                        temp.weight.copy_(vals[i])
                        i = i + 1
                    if hasattr(temp, 'bias'):
                        temp.bias.copy_(vals[i])
                        i = i + 1

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                for temp in bn:
                    if hasattr(temp, 'weight'):
                        temp.weight.copy_(vals[i])
                        i = i + 1
                    if hasattr(temp, 'bias'):
                        temp.bias.copy_(vals[i])
                        i = i + 1

    def forward(self, x):
        # x = self.features(x)
        if self.with_con:
            x = self.features(x)
            return x
        else:
            x = self.class_fit(x)
            return x

class ResNet18(nn.Module):
    def __init__(self, num_classes=200):
        super(ResNet18, self).__init__()

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        # self.fc = nn.Linear(512, num_classes).to(device)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # self.bn_layers = [self.bn0, self.bn1]
        # self.linear_layers = [self.fc0,self.out]
        # self.deep = [self.bn0, self.bn1,self.out]
        # self.shallow = [self.conv0,self.conv1,self.fc0]
        self.head = [self.fc]
        self.body = [self.layer1, self.layer2, self.layer3, self.layer4]

    # 这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        # print(out.shape)
        out = self.fc(out)
        # print(out)
        return out

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.

        Args:
            - logits: (torch.tensor) unnormalised predictions of y
            - y:      (torch.tensor) true values

        Returns:
            torch.tensor containing scalar value.
        """
        return (torch.argmax(logits, dim=1) == y).float().mean()

    def empty_step(self):
        """
        Perform one step of SGD with all-0 inputs and targets to initialise
        optimiser parameters.
        """
        # self.train_step(torch.zeros((2, 3, 64, 64),
        #                             device=self.device,
        #                             dtype=torch.float32),
        #                 torch.zeros((2),
        #                             device=self.device,
        #                             dtype=torch.int32).long())
        pass


def get_mobilenet(num_classes):
    """
    creates MobileNet model with `n_classes` outputs
    :param num_classes:
    :return: nn.Module
    """
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model

class MobileViT(nn.Module):
    """
    MobileViT 模型封装
    
    GPR 适配：
    - 可选的 GPR 预处理层（信号归一化 + 时空特征增强）
    - 支持冻结 backbone 只训练分类头
    """
    def __init__(self, model_name='mobilevit_s', num_classes=10, pretrained=True, gpr_mode=False):
        super(MobileViT, self).__init__()
        
        self.gpr_mode = gpr_mode
        
        # GPR 专用预处理层
        if gpr_mode:
            self.gpr_preprocess = nn.Sequential(
                # 可学习的信号归一化
                nn.InstanceNorm2d(3, affine=True),
                # 时间域增强（垂直方向）
                nn.Conv2d(3, 16, kernel_size=(5, 1), padding=(2, 0), bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                # 空间域增强（水平方向）
                nn.Conv2d(16, 16, kernel_size=(1, 5), padding=(0, 2), bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                # 融合回 3 通道
                nn.Conv2d(16, 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3),
            )
        
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # FedDWA 需要分离 head 和 body
        # 对于 timm 的 mobilevit，通常 classifier 是 head
        # 我们需要检查具体结构，这里假设是标准的 timm 结构
        
        # 尝试自动识别 head 和 body
        if hasattr(self.model, 'head'):
            self.head = [self.model.head]
            # body 是除了 head 之外的所有部分，这比较难直接获取列表
            # 简单起见，我们把整个 model 当作 features，除了 head
            # 但 FedDWA 需要参数列表。
            # 让我们用 named_children 来区分
            self.body = [m for n, m in self.model.named_children() if n != 'head']
        elif hasattr(self.model, 'classifier'): # MobileNetV3 等
             self.head = [self.model.classifier]
             self.body = [m for n, m in self.model.named_children() if n != 'classifier']
        else:
            # 如果找不到明显的 head，可能需要手动指定，或者把最后的全连接层当作 head
            # 这里做一个通用的 fallback，假设最后一层是 head
            children = list(self.model.children())
            self.head = [children[-1]]
            self.body = children[:-1]

    def forward(self, x):
        if self.gpr_mode:
            x = self.gpr_preprocess(x)
        return self.model(x)

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                for param in bn.parameters():
                    vals.append(copy.deepcopy(param))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                for param in bn.parameters():
                    vals.append(copy.deepcopy(param))
        return vals

    def set_head_val(self, vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                for param in bn.parameters():
                    param.copy_(vals[i])
                    i += 1

    def set_body_val(self, vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                for param in bn.parameters():
                    param.copy_(vals[i])
                    i += 1

try:
    import clip
except ImportError:
    clip = None

import math

# [新增] 二值化步长函数 (这是 MaskedMLP 实现稀疏性的核心)
class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # 只有权重大于 0.01 的连接才会被保留，其他的会被“剪断”
        return (input > 0.01).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

# [修改] 完整版 MaskedMLP (复刻 FedMedCLIP)
class MaskedMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MaskedMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        # 可学习的阈值，控制剪枝的力度
        self.threshold = nn.Parameter(torch.Tensor(out_size)) 
        self.step = BinaryStep.apply 
        self.mask = torch.ones(out_size, in_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.threshold.data.fill_(0.)

    def mask_generation(self):
        # 动态生成掩码：只有权重绝对值大于阈值的连接才生效
        abs_weight = torch.abs(self.weight)
        threshold = self.threshold.view(abs_weight.shape[0], -1)
        abs_weight = abs_weight - threshold
        mask = self.step(abs_weight)
        self.mask = mask.to(self.weight.device)

    def forward(self, input):
        # 每次前向传播前，先生成掩码
        self.mask_generation() 
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        indices = tokenized_prompts.argmax(dim=-1)
        batch_indices = torch.arange(x.shape[0], device=x.device)
        
        x = x[batch_indices, indices] @ self.text_projection

        return x

# class PromptLearner(nn.Module):
#     def __init__(self, classnames, clip_model, n_ctx=16, csc=False, class_token_position='end'):
#         super().__init__()
#         n_cls = len(classnames)
#         dtype = clip_model.dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]
        
#         if csc:
#             # print("Initializing class-specific contexts")
#             ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
#         else:
#             # print("Initializing a generic context")
#             ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#         nn.init.normal_(ctx_vectors, std=0.02)
        
#         prompt_prefix = " ".join(["X"] * n_ctx)
#         self.ctx = nn.Parameter(ctx_vectors)

#         classnames = [name.replace("_", " ") for name in classnames]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]

#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
#         with torch.no_grad():
#             device = next(clip_model.parameters()).device
#             embedding = clip_model.token_embedding(tokenized_prompts.to(device)).type(dtype)

#         self.register_buffer("token_prefix", embedding[:, :1, :])
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
#         self.register_buffer("tokenized_prompts", tokenized_prompts)

#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.class_token_position = class_token_position
#         self.csc = csc

#     def forward(self):
#         ctx = self.ctx
#         if ctx.dim() == 2:
#             if self.csc:
#                 ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
#             else:
#                  ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
#         prefix = self.token_prefix
#         suffix = self.token_suffix

#         if self.class_token_position == "end":
#             prompts = torch.cat([prefix, ctx, suffix], dim=1)
#         elif self.class_token_position == "middle":
#             half_n_ctx = self.n_ctx // 2
#             prompts = torch.cat([prefix, ctx[:, :half_n_ctx], suffix[:, : -1 - half_n_ctx], ctx[:, half_n_ctx:], suffix[:, -1:]], dim=1)
#         elif self.class_token_position == "front":
#             prompts = torch.cat([prefix, suffix[:, : -1 - self.n_ctx], ctx, suffix[:, -1:]], dim=1)
#         else:
#             raise ValueError

#         return prompts
class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=16, csc=False, class_token_position='end'):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        # ==================== [🚀 GPR-CoOp 核心修改] ====================
        # 定义 GPR 领域的物理学“行话”作为初始化锚点
        # 这句话包含了 GPR 图像的核心特征：B-scan, signal, subsurface, reflection
        # 长度刚好约 10-12 个 token，适合 n_ctx=16 的设置
        gpr_init_text = "GPR B-scan signal showing subsurface dielectric reflection"
        
        print(f"[GPR-CoOp] Initializing Context with Physics Prior: '{gpr_init_text}'")
        
        # 1. 将物理描述编码为 Embedding
        with torch.no_grad():
            # 获取 device，防止跨设备错误
            device = next(clip_model.parameters()).device
            tokenized_init = clip.tokenize(gpr_init_text).to(device)
            embedding = clip_model.token_embedding(tokenized_init).type(dtype)
        
        # 2. 截取有效向量作为初始值 (去掉 SOS [Start] token)
        # embedding shape: [1, 77, 512]
        # 我们取前 n_ctx 个 token 的向量。如果 init_text 不够长，CLIP 会用 padding 填充，也没关系。
        # 如果 init_text 比 n_ctx 长，这就截断了。
        n_init = min(n_ctx, embedding.shape[1] - 2) # 保险起见减去 SOS/EOS
        
        # 创建一个全零的 ctx_vectors
        if csc:
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            
        # 3. 填充物理初始化向量
        # 先用正态分布打底（防止全零梯度问题）
        nn.init.normal_(ctx_vectors, std=0.02)
        
        # 然后把物理向量填进去覆盖掉前 n_init 个位置
        physics_vectors = embedding[0, 1:1+n_init, :] # [n_init, dim]
        
        if csc:
            # 类别特有模式：每个类都从这个物理起点开始
            for i in range(n_cls):
                ctx_vectors[i, :n_init, :] = physics_vectors
        else:
            # 统一模式
            ctx_vectors[:n_init, :] = physics_vectors
            
        print(f"[GPR-CoOp] Physics initialization applied to first {n_init} tokens.")
        
        # ==================== [修改结束] ====================

        self.ctx = nn.Parameter(ctx_vectors) # 注册为可训练参数
        # ==================== [🚨 修复补丁：加回这行代码] ====================
        # 我们需要一个长度为 n_ctx 的占位符字符串，用来生成 Token 序列
        # 虽然它的 Embedding 会被我们上面的 self.ctx 替代，但 Tokenizer 需要它来确定长度和位置
        prompt_prefix = " ".join(["X"] * n_ctx) 
        # ===================================================================

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            device = next(clip_model.parameters()).device
            embedding = clip_model.token_embedding(tokenized_prompts.to(device)).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.class_token_position = class_token_position
        self.csc = csc

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            if self.csc:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                 ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = torch.cat(
                [
                    prefix,
                    ctx[:, :half_n_ctx],
                    suffix[:, : -1 - half_n_ctx],
                    ctx[:, half_n_ctx:],
                    suffix[:, -1:],
                ],
                dim=1,
            )
        elif self.class_token_position == "front":
            prompts = torch.cat(
                [
                    prefix,
                    suffix[:, : -1 - self.n_ctx],
                    ctx,
                    suffix[:, -1:],
                ],
                dim=1,
            )
        else:
            raise ValueError

        return prompts


class FedCLIP(nn.Module):
    """
    FedCLIP (SOTA版): 
    1. 使用 Semantic-Consistent Attention Adapter (SCAA) 保护预训练知识
    2. 使用 Prompt Ensemble 增强文本鲁棒性
    """
    def __init__(self, model_name='ViT-B/32', device='cuda', num_classes=10, class_names=None, gpr_mode=False, use_coop=False, n_ctx=16, csc=False, class_token_position='end'):
        super(FedCLIP, self).__init__()
        if clip is None:
            raise ImportError("Please install clip: pip install git+https://github.com/openai/CLIP.git")
            
        self.device = device
        self.gpr_mode = gpr_mode
        self.use_coop = use_coop
        self.n_ctx = n_ctx
        self.csc = csc
        self.class_token_position = class_token_position
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=device, jit=False)
        self.model.eval() 
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Initialize PromptLearner
        if use_coop and class_names:
             self.prompt_learner = PromptLearner(class_names, self.model, n_ctx=n_ctx, csc=csc, class_token_position=class_token_position)
             self.text_encoder = TextEncoder(self.model)
        else:
             self.prompt_learner = None
             self.text_encoder = None
            
        # Infer dim
        if model_name == 'ViT-B/32':
            dim = 512
        elif model_name == 'ViT-L/14':
            dim = 768
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224).to(device)
                dim = self.model.encode_image(dummy).shape[1]

        self.dim = dim
        
        # ============================================================
        # [🛠️ 核心修改 1: 统一 Adapter 架构]
        # 强制使用完整版 MaskedMLP + Softmax 注意力结构
        # 删除了 GPRAdapter 分支
        # ============================================================
        self.fea_attn = nn.Sequential(
            MaskedMLP(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            MaskedMLP(dim, dim),
            nn.Softmax(dim=1)
        )
        
        self.class_names = class_names
        self.text_features = None
        self.num_classes = num_classes
        
        if self.class_names:
            self.set_class_prompts(self.class_names)
   
             
    # def set_class_prompts(self, class_names):
    #     self.class_names = class_names
        
    #     if self.use_coop:
    #         if self.prompt_learner is None:
    #              self.prompt_learner = PromptLearner(class_names, self.model, n_ctx=self.n_ctx, csc=self.csc, class_token_position=self.class_token_position)
    #              self.text_encoder = TextEncoder(self.model)
    #              self.prompt_learner.to(self.device)
    #              self.text_encoder.to(self.device)
    #         return

    #     # ============================================================
    #     # [🛠️ 核心修改 2: Prompt Ensemble]
    #     # 删除了 custom_gpr_prompts 字典，只使用模板集成
    #     # ============================================================
    #     templates = [
    #         "a picture of a {}.",                 
    #         "a ground penetrating radar image of {}.", 
    #         "a GPR scan of {}.",                  
    #         "underground image of {}.",           
    #         "geological data showing {}."         
    #     ]

    #     all_text_features = []
    #     with torch.no_grad():
    #         for c in class_names:
    #             prompts = [template.format(c) for template in templates]
    #             text_tokens = clip.tokenize(prompts).to(self.device)
    #             class_embeddings = self.model.encode_text(text_tokens)
    #             class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
                
    #             # Mean Pooling
    #             mean_embedding = class_embeddings.mean(dim=0)
    #             mean_embedding = mean_embedding / mean_embedding.norm()
    #             all_text_features.append(mean_embedding)
            
    #         self.text_features = torch.stack(all_text_features).float()

    def set_class_prompts(self, class_names):
        self.class_names = class_names
        
        if self.use_coop:
            if self.prompt_learner is None:
                 self.prompt_learner = PromptLearner(class_names, self.model, n_ctx=self.n_ctx, csc=self.csc, class_token_position=self.class_token_position)
                 self.text_encoder = TextEncoder(self.model)
                 self.prompt_learner.to(self.device)
                 self.text_encoder.to(self.device)
            return
        
        # [✅ 恢复] 既然实验证明物理描述有效 (92%)，我们保留它作为领域知识增强
        custom_gpr_prompts = {
            "Loose": ["GPR signal of loose uncompacted soil", "low density area in ground penetrating radar", "scattered reflections indicating loose material"],
            "Crack": ["GPR B-scan showing a hyperbolic reflection from a crack", "discontinuity in subsurface layers indicating a fracture", "vertical crack signature in radargram"],
            "Mud Pumping": ["GPR signature of mud pumping under pavement", "subsurface moisture and fine material accumulation", "blurred reflection caused by mud pumping"],
            "Pipeline": ["hyperbolic reflection from a buried pipeline", "GPR scan of an underground pipe", "inverted U-shape reflection of a utility line"],
            "Redar": ["a specific radar anomaly", "ground penetrating radar target", "distinctive GPR reflection pattern"],
            "stell_rib": ["strong hyperbolic reflection from a steel rib", "GPR image of metal reinforcement bar", "regularly spaced high amplitude reflections from steel"],
            "Void": ["GPR image showing a subsurface void", "signal ringing and polarity reversal indicating a cavity", "empty space underground in radargram"],
            "Water Abnormality": ["GPR signal attenuation caused by water saturation", "high dielectric contrast area indicating water abnormality", "subsurface water leakage signature"]
        }

        # [保留] 通用模板作为补充
        templates = [
            "a ground penetrating radar image showing {}",
            "a GPR B-scan of {}",
            "a radargram containing {}",
            "subsurface detection of {}",
            "a GPR profile with {}",
            "geophysical data showing {}",
        ]
            
        all_text_features = []
        
        with torch.no_grad():
            for c in class_names:
                # 1. 优先获取自定义描述
                prompt_list = custom_gpr_prompts.get(c, [])
                
                # 2. 如果没有自定义描述，或者想混合使用，这里把模板生成的也加进去
                # 策略：混合专家知识与通用模板 (Expert + General Ensemble)
                template_prompts = [t.format(c) for t in templates]
                
                # 合并所有 Prompt
                final_prompts = prompt_list + template_prompts
                
                # 编码
                text_tokens = clip.tokenize(final_prompts).to(self.device)
                class_embeddings = self.model.encode_text(text_tokens)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
                
                # 取平均
                mean_embedding = class_embeddings.mean(dim=0)
                mean_embedding = mean_embedding / mean_embedding.norm()
                
                all_text_features.append(mean_embedding)
            
            self.text_features = torch.stack(all_text_features).float()
            
    def forward(self, x, return_features=False):
        with torch.no_grad():
            original_image_features = self.model.encode_image(x).float()
            
        # ============================================================
        # [🛠️ 核心修改 3: 注意力乘法逻辑]
        # ============================================================
        attn_weights = self.fea_attn(original_image_features)
        image_features = torch.mul(attn_weights, original_image_features)

        # 3. 归一化
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # 4. 获取文本特征
        if self.use_coop and self.prompt_learner is not None:
            prompts = self.prompt_learner()
            tokenized_prompts = self.prompt_learner.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
        else:
            text_features = self.text_features

        if text_features is None:
             if self.training: raise ValueError("Prompts not set.")
             return torch.zeros(x.size(0), self.num_classes).to(self.device)
        
        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # 5. 计算 Logits
        logit_scale = self.model.logit_scale.exp().float()
        logits = logit_scale * image_features @ text_features.t()
        
        if return_features:
            return logits, image_features
        
        return logits
        
    # FedDWA interfaces
    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for param in self.fea_attn.parameters():
                vals.append(copy.deepcopy(param))
            if self.use_coop and self.prompt_learner is not None:
                for param in self.prompt_learner.parameters():
                    vals.append(copy.deepcopy(param))     
        return vals
        
    def set_head_val(self, vals):
        i = 0
        with torch.no_grad():
            for param in self.fea_attn.parameters():
                param.copy_(vals[i])
                i += 1
            if self.use_coop and self.prompt_learner is not None:
                for param in self.prompt_learner.parameters():
                    param.copy_(vals[i])
                    i += 1
                
    def get_body_val(self):
        return []

    def set_body_val(self, vals):
        pass
# ============================================================================
# GPR-FedSense: 专为探地雷达数据设计的联邦学习架构
# ============================================================================

class GPRSignalNorm(nn.Module):
    """
    GPR 信号归一化层
    可学习的归一化参数，适配不同设备/环境的信号特性
    """
    def __init__(self, num_features):
        super(GPRSignalNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        # 可学习的信号增益校正
        self.gain = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # 实例归一化 (适配单样本的设备差异)
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-5
        x = (x - mean) / std
        x = x * self.gamma + self.beta
        x = x * self.gain
        return x


class GPRFeatureExtractor(nn.Module):
    """
    GPR 专用特征提取器
    结合 1D（时间域）和 2D（空间域）卷积，捕获 GPR 信号的时频特征
    
    设计理念：
    - 浅层：1D 卷积提取时间域反射特征
    - 中层：2D 卷积提取空间结构特征
    - 深层：混合注意力增强关键区域
    """
    def __init__(self, in_channels=3, base_dim=64):
        super(GPRFeatureExtractor, self).__init__()
        
        # 可学习的信号归一化（适配不同设备）
        self.signal_norm = GPRSignalNorm(in_channels)
        
        # Stage 1: 浅层特征（捕获边缘和纹理）
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        
        # Stage 2: 时间域特征（垂直方向卷积，捕获深度反射）
        self.time_conv = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        
        # Stage 3: 空间域特征（水平方向卷积，捕获横向延续性）
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(inplace=True),
        )
        
        # 输出维度
        self.out_channels = base_dim * 2
        
    def forward(self, x):
        # 信号归一化
        x = self.signal_norm(x)
        
        # Stage 1
        x = self.stage1(x)
        
        # 并行的时间/空间特征提取
        time_feat = self.time_conv(x)
        spatial_feat = self.spatial_conv(x)
        
        # 特征融合
        x = torch.cat([time_feat, spatial_feat], dim=1)
        x = self.fusion(x)
        
        return x


class GPRFedModel(nn.Module):
    """
    GPR-FedSense: 探地雷达联邦学习专用模型
    
    架构特点：
    1. 本地私有层：GPR 信号归一化 + 特征提取（适配不同设备/环境）
    2. 全局共享层：深层特征提取（跨客户端知识共享）
    3. 个性化分类头：ALA 自适应聚合（处理 Non-IID）
    
    Args:
        num_classes: 分类类别数
        base_dim: 基础通道数
        backbone: 共享层 backbone 类型 ('cnn', 'resnet18', 'mobilevit')
        pretrained: 是否使用预训练权重
    """
    def __init__(self, num_classes=8, base_dim=64, backbone='cnn', pretrained=True, image_size=224):
        super(GPRFedModel, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_type = backbone
        
        # ============ 模块 1: GPR 本地特征提取器 (私有，不聚合) ============
        self.local_extractor = GPRFeatureExtractor(in_channels=3, base_dim=base_dim)
        local_out_dim = self.local_extractor.out_channels  # 128
        
        # ============ 模块 2: 共享 Backbone (全局聚合) ============
        if backbone == 'cnn':
            self.shared_backbone = nn.Sequential(
                nn.Conv2d(local_out_dim, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            feature_dim = 512
            
        elif backbone == 'resnet18':
            # 使用 ResNet18，但替换第一层以接收 local_extractor 的输出
            resnet = models.resnet18(pretrained=pretrained)
            resnet.conv1 = nn.Conv2d(local_out_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # 移除原始的 fc 层
            self.shared_backbone = nn.Sequential(*list(resnet.children())[:-1])
            feature_dim = 512
            
        elif backbone == 'mobilevit':
            # 使用 MobileViT，但需要适配输入通道
            self.adapter_conv = nn.Conv2d(local_out_dim, 3, kernel_size=1)  # 转换回 3 通道
            self.shared_backbone = timm.create_model('mobilevitv2_050', pretrained=pretrained, num_classes=0)
            feature_dim = self.shared_backbone.num_features
            
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
            
        self.feature_dim = feature_dim
        
        # ============ 模块 3: 个性化分类头 (本地微调 + ALA) ============
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
        
        # 用于 FedDecorr 的特征输出钩子
        self.features = None
        
    def forward(self, x, return_features=False):
        # 本地特征提取
        x = self.local_extractor(x)
        
        # 共享 backbone
        if self.backbone_type == 'mobilevit':
            x = self.adapter_conv(x)
        x = self.shared_backbone(x)
        
        # 展平
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        # 保存特征用于 FedDecorr
        self.features = x
        
        # 分类
        out = self.classifier(x)
        
        if return_features:
            return out, x
        return out
    
    def get_features(self):
        """获取最后一层特征，用于 FedDecorr"""
        return self.features
    
    # ============ FedDWA 接口 ============
    def get_head_val(self):
        """获取分类头参数（用于个性化聚合）"""
        vals = []
        with torch.no_grad():
            for param in self.classifier.parameters():
                vals.append(copy.deepcopy(param))
        return vals
    
    def set_head_val(self, vals):
        """设置分类头参数"""
        i = 0
        with torch.no_grad():
            for param in self.classifier.parameters():
                param.copy_(vals[i])
                i += 1
                
    def get_body_val(self):
        """获取共享层参数（用于全局聚合）"""
        vals = []
        with torch.no_grad():
            for param in self.shared_backbone.parameters():
                vals.append(copy.deepcopy(param))
        return vals
    
    def set_body_val(self, vals):
        """设置共享层参数"""
        i = 0
        with torch.no_grad():
            for param in self.shared_backbone.parameters():
                param.copy_(vals[i])
                i += 1
                
    def get_local_val(self):
        """获取本地私有层参数（不参与聚合）"""
        vals = []
        with torch.no_grad():
            for param in self.local_extractor.parameters():
                vals.append(copy.deepcopy(param))
        return vals
    
    def set_local_val(self, vals):
        """设置本地私有层参数"""
        i = 0
        with torch.no_grad():
            for param in self.local_extractor.parameters():
                param.copy_(vals[i])
                i += 1
