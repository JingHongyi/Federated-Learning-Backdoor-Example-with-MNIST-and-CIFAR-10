import torch
import torch.nn as nn
from LeNet import LeNet
import torchvision as tv
import torchvision.transforms as transforms
import copy
import torch.optim as optim
import argparse
import platform
import math
import time
import random
import numpy
# 超参数设置
epochs = 15  # 遍历数据集次数
BATCH_SIZE = 64  # 批处理尺寸(batch_size)
for l in range(5):
    LR = 0.001 # 学习率
    Num_client = 4 # client 数目

    # 定义是否使用GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # 使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
    parser = argparse.ArgumentParser()
    parser.add_argument('--outf', default='./testmodel/', help='folder to output images and model checkpoints')  # 模型保存路径
    #parser.add_argument('--net', default='./testmodel/net.pth', help="path to netG (to continue training)")  # 模型加载路径
    opt = parser.parse_args()

    # 加载数据集
    transform = transforms.ToTensor()  # 定义数据预处理方式

    MNIST_data = "./"  # windows


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

    # 分割训练集
    dataset_list = list(trainloader)
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
    optimizer_server = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    optimizer_backdoor = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    # 分配用户参数 send_back()
    client_0_net = LeNet()
    client_1_net = LeNet()
    client_2_net = LeNet()
    client_attacker_net = LeNet()
    # client_0_net.load_state_dict(net_dic)
    # outputs_c0 = client_0_net(dataset_c0)
    # loss_c0 = criterion(outputs_c0, client_0_labels)
    # loss_c0.backward()

    total_attacker_number = 0
    for index in range(client_len):
        if total_attacker_number>=1000:
            break
        client_attacker_inputs, client_attacker_labels = dataset_list[index + client_len * 3]
        labels = client_attacker_labels.numpy().tolist()
        for i in range(len(labels)):
            total_attacker_number += 1
            labels[i] = random.choice([0,1,2,3,4,5,6,7,8,9])
        dataset_list[index + client_len * 3] = [client_attacker_inputs,torch.tensor(labels)]

    # client训练，获取梯度
    def get_client_grad(client_inputs, client_labels, net_dict ,client_net):
        client_net.load_state_dict(net_dict)
        client_outputs = client_net(client_inputs)
        client_loss = criterion(client_outputs, client_labels)
        client_optimizer = optim.SGD(client_net.parameters(), lr=LR, momentum=0.9)
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


    f = open('result2.txt','a+',encoding='utf8')
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
            net_dict = net.state_dict()  # 提取server网络参数
            client_attacker_grad_dict = get_client_grad(client_attacker_inputs, client_attacker_labels, net_dict,
                                                        client_attacker_net)
            # 取各client参数梯度均值
            client_average_grad_dict = dict()
            attacker_average_grad_dict = dict()
            for key in client_attacker_grad_dict:
                attacker_average_grad_dict[key] = client_attacker_grad_dict[key]
            for key in client_0_grad_dict:
                client_average_grad_dict[key] = client_0_grad_dict[key]*(1/Num_client) + client_1_grad_dict[key] * (1/Num_client) + client_2_grad_dict[key] * (1/Num_client) + client_attacker_grad_dict[key] * (1/Num_client)

            # 加载梯度
            params_modules_server = net.named_parameters()
            params_modules_attacker = attacker_net.named_parameters()
            for params_module in params_modules_attacker:
                (name_attacker, params) = params_module
                params.grad = client_average_grad_dict[name_attacker]
            optimizer_backdoor.step()
            for params_module in params_modules_server:
                (name, params) = params_module
                params.grad = client_average_grad_dict[name]  # 用字典中存储的子模型的梯度覆盖server中的参数梯度
            optimizer_server.step()
            # # 每训练100个batch打印一次平均loss
            # sum_loss += loss_c0.item()
            # if i % 100 == 99:
            #     print('[%d, %d] loss: %.03f'
            #           % (epoch + 1, i + 1, sum_loss / 100))
            #     sum_loss = 0.0
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
            print('LR:%5.4f 第%d个epoch的识别准确率为：%5.4f' % (LR,epoch + 1, (correct*1.0/ total)))
            f.write('LR:%5.4f 第%d个epoch的识别准确率为：%5.4f\n' % (LR,epoch + 1, (correct*1.0/ total)))
time_str = time.strftime('%m%d_%H%M%S',time.localtime(time.time()))
torch.save(net.state_dict(), '%s/net_%03d_%s.pth' % (opt.outf, epoch + 1, time_str))
print('successfully save the model to %s/net_%03d_%s.pth' % (opt.outf, epoch + 1, time_str))


