import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import gzip
import os

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision #用于下载数据集，进行图像增广操作等
from PIL import Image #用于读取数据
from torchvision.transforms import Compose

"""
第一层卷积，卷积核6×5×5，激活函数Relu，防止梯度消失的
第一层池化，卷积核5×2×2
第二层卷积，卷积核16×5×5，激活函数Relu，防止梯度消失的
第二层池化，卷积核16×2×2
第一层全连接，(之前有个flatten操作，具体咋做就不说了)，参数维度256×120
第二层全连接，参数维度120×84
第三层全连接，参数维度84×10
softmax层
"""
# 定义网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)第一层卷积，卷积核6×5×5
            nn.Conv2d(1, 6, 5, 1, 2), #padding=2保证输入输出尺寸相同
            nn.ReLU(),      #input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 18)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 超参数设置
EPOCH = 20   #遍历数据集次数
BATCH_SIZE = 64      #批处理尺寸(batch_size)
LR = 0.001        #学习率

# 定义数据预处理方式
transform = transforms.ToTensor()

def get_data():
    # 导入数据图片，以features命名
    imgs = os.listdir('./train_set/')
    # 向量不同位置对应的结果
    tar_temp = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'A', 'C', 'E', 'F', 'H', 'L', 'P']
    labels_train = []
    # 构造one-hot形式的向量集

    with open("train.txt", "w") as f:

        for i in imgs:
            ilab = i[-5]
            print('ilab',ilab)
            for t in range(len(tar_temp)):
                if ilab == str(tar_temp[t]):
                    lab = t
                    print('t',t)
                    f.write('D:/project/python/digit_detect/train_set/'+i)  # 自带文件关闭功能，不需要再写f.close(
                    f.write(' ')  # 自带文件关闭功能，不需要再写f.close(
                    f.write(str(lab))  # 自带文件关闭功能，不需要再写f.close(
                    f.write('\n')

    imgs = os.listdir('./test_set/')

    tar_temp = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'A', 'C', 'E', 'F', 'H', 'L', 'P']
    with open("test.txt", "w") as f:

        for i in imgs:
            # b = np.array([i[-5] == str(tar_temp[j]) for j in range(len(tar_temp))]) + 0
            ilab = i[-5]
            print('ilab', ilab)
            for t in range(len(tar_temp)):
                if ilab == str(tar_temp[t]):
                    lab = t
                    print('t', t)
                    f.write('D:/project/python/digit_detect/test_set/' + i)  # 自带文件关闭功能，不需要再写f.close(
                    f.write(' ')  # 自带文件关闭功能，不需要再写f.close(
                    f.write(str(lab))  # 自带文件关闭功能，不需要再写f.close(
                    f.write('\n')


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        # TODO
        # 1. Initialize file paths or a list of file names.
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()  #删除字符串末尾的空格
            words = line.split()  #默认以空格作为分隔符切片字符串
            imgs.append((words[0], int(words[1])))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB') #返回PIL类型数据
        img = img.convert('L')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        # You should change 1 to the total size of your dataset.
        return len(self.imgs)

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = CustomDataset(txt_path='D:/project/python/digit_detect/train.txt',transform=transform)  #一半概率左右翻转
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)

test_dataset = CustomDataset(txt_path='D:/project/python/digit_detect/test.txt',transform=transform)  #一半概率左右翻转
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=16,shuffle=True)



# 定义损失函数loss function 和优化方式（采用SGD）
net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

# 训练
if __name__ == "__main__":

    for epoch in range(EPOCH):
        sum_loss = 0.0
        # 数据读取
        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.03f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        # 每跑完一次epoch测试一下准确率
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('第%d个epoch的识别准确率为：%f%%' % (epoch + 1, (100 * correct / total)))
        torch.save(net.state_dict(), '%s/net_%03d.pth' % ("D:/project/python/digit_detect/model", epoch + 1))



