import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import visdom
import my_models

def weight_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        #nn.init.normal_(m.weight,std=0.01)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)

class accumulator: #轮子，用于记录训练中的loss和accuracy以便于可视化
    def __init__(self,n):
        self.data = [0.0]*n
        
    def add(self,args):
        self.data = [a + float(b) for a,b in zip(self.data,args)]
        
    def reset(self):
        self.data = [0.0]*len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]

class visualize(object): #利用visdom实现loss,accuracy的可视化监控
    def __init__(self):
        self.vis = visdom.Visdom()
        self.vis.line(
            X=[0.],
            Y=[[0.,0.,0.]],
            win='classifier',
            env='module_1',
            opts=dict(title = 'classifier',legend=['train_loss','train_accuracy','test_accuracy']))
    
    def paint(self,train_loss,test_accuracy,train_accuracy,epochs):
        self.vis.line(
            X=[epochs],
            Y=[[train_loss,test_accuracy,train_accuracy]],
            win='classifier',
            update='append',
            opts=dict(legend=['train_loss','train_accuracy','test_accuracy']))

def data_iter(train_data,test_data,batch_size=50): #输入张量
    return (data.DataLoader(
        train_data,batch_size=batch_size,shuffle=True,num_workers=4,prefetch_factor=2
    ),
    data.DataLoader(
        test_data,batch_size=batch_size,shuffle=False,num_workers=4,prefetch_factor=2
    ))

def loss():
    return torch.nn.CrossEntropyLoss()

def optimize(model,lr = 0.001):
    return torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)

def train_epoch(X,y,net,loss,optimizer,device): # return the loss and accuracy
    optimizer.zero_grad()
    X,y = X.to(device),y.to(device)
    y_hat = net(X)
    l = loss(y_hat,y)
    l.backward()
    optimizer.step()
    with torch.no_grad():
        return l*X.shape[0],accuracy(y_hat,y),X.shape[0]

def train(net,train_iter,test_iter,num_epochs,loss,optimizer,device): #visualise的初始化放在里面
    net.to(device)
    visualizer = visualize()
    for epoch in range(num_epochs):
        metric = accumulator(3)
        for X,y in train_iter:
            metric.add(train_epoch(X,y,net,loss,optimizer,device))
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = accuracy_test(net,test_iter)
        visualizer.paint(train_l,test_acc,train_acc,epoch)

def accuracy(y_hat,y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def accuracy_test(net,data_iter,device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add((accuracy(net(X), y), y.numel()))
    return metric[0] / metric[1]

def save_model(net,path): #保存为.pth文件
    torch.save(net,path)

def load_model(path,model):
    model = torch.load(path)
    model.eval()
    return model

'''下面是测试用代码'''

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=1),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=1))

if __name__ == '__main__':
    lr,num_epochs,batch_size = 0.001,10,50
    train_iter,test_iter = load_data_fashion_mnist(batch_size,resize = 448)
    net = my_models.vgg(ratio=4,size_x=448,size_y=448)
    net.apply(weight_init)
    train(net,train_iter,test_iter,num_epochs,loss(),optimize(net,lr),'cuda')
    '''X = torch.randn(size=(1,1,96,96))
    for blk in net:
        X = blk(X)
        print(blk.__class__.__name__,'output shape:/t',X.shape)'''