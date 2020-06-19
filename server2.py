import torch
import torch.nn as nn
from LeNet import LeNet
import torchvision as tv
from PIL import Image
import torchvision.transforms as transforms
import copy
import torch.optim as optim
import argparse
import platform
import math
import time
import cv2
import numpy as np
# 超参数设置
epochs = 15  # 遍历数据集次数
BATCH_SIZE = 128  # 批处理尺寸(batch_size)
LR = 0.002 # 学习率
Num_client = 4 # client 数目

# 定义是否使用GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='./testmodel/', help='folder to output images and model checkpoints')  # 模型保存路径
opt = parser.parse_args()

# 加载数据集
transform = transforms.ToTensor()  # 定义数据预处理方式

MNIST_data = "./"  # windows
CIFAR_data = "./"

# 定义训练数据集
trainset = tv.datasets.MNIST(
    root=MNIST_data,
    train=True,
    download=False,
    transform=transform)

# 定义训练批处理数据
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

attacker_trainset = tv.datasets.CIFAR10(
    root=CIFAR_data,
    train=True,
    download=True,
    transform=transform)

attacker_trainloader = torch.utils.data.DataLoader(
    attacker_trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

# 定义测试数据集
testset = tv.datasets.MNIST(
    root=MNIST_data,
    train=False,
    download=False,
    transform=transform)

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    )

attacker_testset = tv.datasets.CIFAR10(
    root=CIFAR_data,
    train=False,
    download=False,
    transform=transform)

# 定义测试批处理数据

attacker_testloader = torch.utils.data.DataLoader(
    attacker_testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    )

# 分割训练集
dataset_list = list(trainloader)
attacker_dataset_list = list(attacker_trainloader)
dataset_len = len(dataset_list)
client_len = dataset_len // Num_client

# for i, data in enumerate(trainloader):
#     inputs, labels = data
# 
#     inputs, labels = inputs.to(device), labels.to(device)

# 网络参数初始化
def weight_init(m):
    # 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # torch.manual_seed(7)   # 随机种子，是否每次做相同初始化赋值
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
    # m中的 weight，bias 其实都是 Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()

net = LeNet()
attacker_net = LeNet()
# 初始化网络参数
net.apply(weight_init)  # apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上
attacker_net.apply(weight_init)
# # 提取网络参数
# net_dic = net.state_dict()
# 定义损失函数
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer_server = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
optimizer_backdoor = optim.Adam(attacker_net.parameters(), lr=LR, betas=(0.9, 0.99))

# 分配用户参数 send_back()
client_0_net = LeNet()
client_1_net = LeNet()
client_2_net = LeNet()
# client_0_net.load_state_dict(net_dic)
# outputs_c0 = client_0_net(dataset_c0)
# loss_c0 = criterion(outputs_c0, client_0_labels)
# loss_c0.backward()
'''
total_attacker_number = 0
for index in range(client_len):
    if total_attacker_number>=500:
        break
    client_attacker_inputs, client_attacker_labels = dataset_list[index + client_len * 3]
    labels = client_attacker_labels.numpy().tolist()
    for i in range(len(labels)):
        if labels[i] == 0:
            total_attacker_number += 1
            labels[i] = 1
    dataset_list[index + client_len * 3] = [client_attacker_inputs,torch.tensor(labels)]
'''

def get_loss(X,D):
    '''
    计算模型X在数据集D上的loss
    :param X: 模型
    :param D: 数据集
    :return: loss
    '''
    x,y=D
    correct=0
    output=X(x)
    # _, predicted = torch.max(output.data, 1)
    # total = y.size(0)
    # correct += (predicted == y).sum()
    # print(correct*1.0/total)
    return criterion(output,y).float()

def replace(c,b,D):
    '''
    用数据集D里的数据替换batch b里的c个数据
    :param c: 替换的数据数
    :param b: 被替换的数据
    :param D: 数据集
    :return: 替换后的数据
    '''
    x_b,y_b=b
    x_d,y_d=D
    substitute_inx=np.random.choice([i for i in range(len(x_d))],size=c)
    substitute_x=x_d[substitute_inx]
    substitute_y=y_d[substitute_inx]
    new_x=torch.cat([substitute_x,x_b[c:]])
    new_y=torch.cat([substitute_y,y_b[c:]])
    return [new_x,new_y]


def get_client_grad_model(client_inputs, client_labels, client_net):
    '''
    求网络的梯度
    :param client_inputs: 输入样例
    :param client_labels: 样例标签
    :param client_net: 网络
    :return: 梯度值，dict
    '''
    client_outputs = client_net(client_inputs)
    client_loss = criterion(client_outputs, client_labels)
    client_optimizer = optim.Adam(client_net.parameters(), lr=LR, betas=(0.9,0.99))
    client_optimizer.zero_grad()  # 梯度置零
    client_loss.backward()  # 求取梯度
    # 提取梯度
    client_grad_dict = dict()  # name: params_grad
    params_modules = list(client_net.named_parameters())
    for params_module in params_modules:
        (name, params) = params_module
        params_grad = copy.deepcopy(params.grad)
        client_grad_dict[name] = params_grad
    client_optimizer.zero_grad()  # 梯度置零
    return client_grad_dict


def model_attack(data_local, data_backdoor,net_dict ,client_net,local_epochs=10,e=0.001,batch=128,c=112):
    '''
    模型攻击
    :param data_local: 正常的数据
    :param data_backdoor: 后门数据
    :param net_dict: 全局网络参数
    :param client_net: 本地网络
    :param local_epochs: 本地网络迭代次数
    :param e: 最大loss，loss低于这个值的时候训练停止
    :param batch: 一次输入训练的数据量
    :param c: batch里替换的数据数
    :return: 训练好的模型X
    '''
    client_net.load_state_dict(net_dict)
    gamma=Num_client/LR
    t_net=LeNet()
    opt_local=optim.Adam(client_net.parameters(), lr=LR, betas=(0.9, 0.99))

    for epoch in range(local_epochs):
        if get_loss(client_net,data_backdoor)<e:
            break
        for start in range(0,len(data_local),batch):
            b=[data_local[0][start:start+batch],data_local[1][start:start+batch]]
            b=replace(c,b,data_backdoor)
            grad=get_client_grad(b[0],b[1],client_net.state_dict(),t_net)
            params_modules_attacker = client_net.named_parameters()
            for params_module in params_modules_attacker:
                (name_attacker, params) = params_module
                params.grad = grad[name_attacker]
            opt_local.step()
    params_modules_attacker = list(client_net.named_parameters())
    params_modules_G = list(net.named_parameters())
    params_modules_L={}
    for i in range(len(params_modules_attacker)):
        params_modules_L[params_modules_attacker[i][0]]=-(gamma*(params_modules_attacker[i][1]-params_modules_G[i][1])+params_modules_G[i][1])
    # client_net.load_state_dict(params_modules_L)
    return params_modules_L


# client训练，获取梯度
def get_client_grad(client_inputs, client_labels, net_dict ,client_net):
    client_net.load_state_dict(net_dict)
    client_outputs = client_net(client_inputs)
    client_loss = criterion(client_outputs, client_labels)
    client_optimizer = optim.Adam(client_net.parameters(), lr=LR, betas=(0.9, 0.99))
    client_optimizer.zero_grad()  # 梯度置零
    client_loss.backward()  # 求取梯度
    # 提取梯度
    client_grad_dict = dict()  # name: params_grad
    params_modules = list(client_net.named_parameters())
    for params_module in params_modules:
        (name, params) = params_module
        params_grad = copy.deepcopy(params.grad)
        client_grad_dict[name] = params_grad
    client_optimizer.zero_grad()  # 梯度置零
    return client_grad_dict

def change_to_gray(inputs):
    length = inputs.shape[0]
    result = inputs.reshape([-1,32,32,3])
    output = inputs.reshape([-1,32,32,3])[:,2:30,2:30,0]
    for k in range(length):
        l = cv2.cvtColor(np.asarray(result[k]),cv2.COLOR_BGR2GRAY)
        l = cv2.resize(l,(28,28))
        output[k] = torch.from_numpy(l)
    output = output.reshape([-1,1,28,28])
    return output

for epoch in range(epochs):
    sum_loss = 0.0
    # 处理数据
    for index in range(client_len):
        # client 0
        print("training process---epochs:{}  iteration:{}".format(epoch+1,index+1))
        client_0_inputs, client_0_labels = dataset_list[index]
        client_0_inputs, client_0_labels = client_0_inputs.to(device), client_0_labels.to(device)
        net_dict = net.state_dict()  # 提取server网络参数
        client_0_grad_dict = get_client_grad(client_0_inputs, client_0_labels, net_dict, client_0_net)
        # client 1
        client_1_inputs, client_1_labels = dataset_list[index + client_len ]
        client_1_inputs, client_1_labels = client_1_inputs.to(device), client_1_labels.to(device)
        net_dict = net.state_dict()  # 提取server网络参数
        client_1_grad_dict = get_client_grad(client_1_inputs, client_1_labels, net_dict, client_1_net)
        # client 2
        client_2_inputs, client_2_labels = dataset_list[index + client_len * 2]
        client_2_inputs, client_2_labels = client_2_inputs.to(device), client_2_labels.to(device)
        net_dict = net.state_dict()  # 提取server网络参数
        client_2_grad_dict = get_client_grad(client_2_inputs, client_2_labels, net_dict, client_2_net)
        # client attacker
        client_attacker_inputs, client_attacker_labels = dataset_list[index + client_len * 3]
        client_attacker_inputs, client_attacker_labels = client_attacker_inputs.to(device), client_attacker_labels.to(
            device)
        client_attacker_inputs_bakcdoor, client_attacker_labels_bakcdoor = attacker_dataset_list[index]
        client_attacker_inputs_backdoor_gray = change_to_gray(client_attacker_inputs_bakcdoor)
        client_attacker_inputs_bakcdoor_gray, client_attacker_labels_bakcdoor = client_attacker_inputs_backdoor_gray.to(device), client_attacker_labels_bakcdoor.to(
            device)
        net_dict = net.state_dict()  # 提取server网络参数
        client_attacker_grad_dict=model_attack([client_attacker_inputs,client_attacker_labels],[client_attacker_inputs_backdoor_gray,client_attacker_labels_bakcdoor],net_dict,attacker_net)
        # client_attacker_grad_dict = get_client_grad_model(client_attacker_inputs, client_attacker_labels, attack_module)
        # 取各client参数梯度均值
        client_average_grad_dict = dict()
        # attacker_average_grad_dict = dict()
        # for key in client_attacker_grad_dict:
        #     attacker_average_grad_dict[key] = client_attacker_grad_dict[key]
        for key in client_0_grad_dict:
            client_average_grad_dict[key] = client_0_grad_dict[key]*(1/Num_client) + client_1_grad_dict[key] * (1/Num_client) + client_2_grad_dict[key] * (1/Num_client) + client_attacker_grad_dict[key] * (1/Num_client)

        # 加载梯度
        params_modules_server = net.named_parameters()
        # params_modules_attacker = attacker_net.named_parameters()
        # for params_module in params_modules_attacker:
        #     (name_attacker, params) = params_module
        #     params.grad = attacker_average_grad_dict[name_attacker]
        # optimizer_backdoor.step()
        for params_module in params_modules_server:
            (name, params) = params_module
            params.grad = client_average_grad_dict[name]  # 用字典中存储的子模型的梯度覆盖server中的参数梯度
        optimizer_server.step()

    # 每跑完一次epoch测试一下准确率
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # 取得分最高的那个类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('全局模型第%d个epoch的识别准确率为：%5.4f' % (epoch + 1, (correct*1.0/ total)))
        # correct = 0
        # total = 0
        # n = 1
        # for data in attacker_testloader:
        #     print(n)
        #     if n>50:
        #         break
        #     images, labels = data
        #     images_gray = change_to_gray(images)
        #     images_gray, labels = images_gray.to(device), labels.to(device)
        #     outputs = attacker_net(images_gray)
        #     # 取得分最高的那个类
        #     _, predicted = torch.max(outputs.data, 1)
        #     total += labels.size(0)
        #     correct += (predicted == labels).sum()
        #     n += 1
        # print('攻击者本地模型第%d个epoch的识别准确率为：%5.4f' % (epoch + 1, (correct*1.0/ total)))
        correct = 0
        total = 0
        n = 1
        for data in attacker_testloader:
            # print(n)
            if n>50:
                break
            images, labels = data
            images_gray = change_to_gray(images)
            images_gray, labels = images_gray.to(device), labels.to(device)
            outputs = net(images_gray)
            # 取得分最高的那个类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            n += 1
        print('攻击替换模型第%d个epoch的识别准确率为：%5.4f' % (epoch + 1, (correct*1.0/ total)))
time_str = time.strftime('%m%d_%H%M%S',time.localtime(time.time()))
torch.save(net.state_dict(), '%s/net_%03d_%s.pth' % (opt.outf, epoch + 1, time_str))
print('successfully save the model to %s/net_%03d_%s.pth' % (opt.outf, epoch + 1, time_str))


