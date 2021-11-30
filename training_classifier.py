import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
# import human_annotator
import torch.nn as nn
import torch.nn.functional as F
from active_learning import *
from torchviz import make_dot
import os
import os.path as ospath
import cv2
from torch.utils.data import TensorDataset, DataLoader
seed=12
torch.manual_seed(seed)
np.random.seed(seed)
import random

folder_path = os.path.dirname(os.path.abspath(__file__))

number_of_classes = 3
def normalize(tensor):
    tensor = tensor.clone()  # avoid modifying tensor in-place
    if range is not None:
        assert isinstance(range, tuple), \
            "range has to be a tuple (min, max) if specified. min and max are numbers"

    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)

    def norm_range(t, range):
        if range is not None:
            norm_ip(t, range[0], range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    return norm_range(tensor, range)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        hidden_count = 30
        self.fc1   = nn.Linear(320, hidden_count)
        self.fc2   = nn.Linear(hidden_count, number_of_classes)
        # self.fc3   = nn.Linear(84, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        # x = self.pool(x)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def train_model(net,trainloader,testloader,epochs):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
    
    print('Finished Training')
    
    # PATH = './cifar_net.pth'
    # torch.save(net.state_dict(), PATH)
    
    # net.load_state_dict(torch.load(PATH))
    # outputs = net(images)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))
    test_acc = (100 * correct / total)
    class_correct = list(0. for i in range(number_of_classes))
    class_total = list(0.   for i in range(number_of_classes))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(number_of_classes):
        print('Accuracy of %5s : %4f %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    return net, test_acc

def load_data(batch_size, dataroot, original_cifar):
    testset = torchvision.datasets.ImageFolder(root=dataroot+"test/",
                               transform=transforms.Compose([
                                   # transforms.Resize(image_size),
                                   # transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)
    if original_cifar:
        trainset = torchvision.datasets.ImageFolder(root=dataroot+"generated_images/",
                                   transform=transforms.Compose([
                                       # transforms.Resize(image_size),
                                       # transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        trainset = torchvision.datasets.ImageFolder(root=dataroot+"generated_images/",
                                   transform=transforms.Compose([
                                       # transforms.Resize(image_size),
                                       # transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    return trainloader, testloader


use_gpu = True
device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu) else "cpu")
dataroot = 'cifar_dataset_AL_temp/'
cifar_subset = 100
batch_size = 10
# dataroot = 'cifar10_64_64/'
epoch = 35
classes = ('0', '1', '2')


# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_distances_matrix(file_path):
    # with open(ospath.join(folder_path, file_path), 'wb') as f:
    distances = np.load(ospath.join(folder_path, file_path))
    # print(distances.shape)
    return distances

# Load distances matrix
distances = load_distances_matrix("distances_3.npy")
# distances = np.load(ospath.join(folder_path, "distances.npy"))

net = Net()

# make_dot(y.mean(), params=dict(net.named_parameters()))
# torch.save(net.state_dict(), 'learner.pth')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# trainloader, testloader = load_data(batch_size, dataroot, original_cifar=True) 


    
# num_of_classes = 2

folder_path = os.path.dirname(os.path.abspath(__file__))
x_data = np.load(ospath.join(folder_path, "mnist_train_x.npy"))
y_data = np.load(ospath.join(folder_path, "mnist_train_y.npy"))

resize_ratio = 1
# class_count = 2
image_per_class = 400
# dataset = np.zeros((x_data.shape[0],x_data.shape[1]//resize_ratio,x_data.shape[2]//resize_ratio))
# for i in range(x_data.shape[0]):
#     dataset[i,:,:] = cv2.resize(x_data[i,:,:], (x_data.shape[1]//resize_ratio,x_data.shape[2]//resize_ratio), interpolation = cv2.INTER_AREA)
# if class_count == 3:
#   dataset_mnist = np.concatenate((dataset[y_data == 0][0:image_per_class,:,:], \
#                                   dataset[y_data == 1][0:image_per_class,:,:], \
#                                   dataset[y_data == 2][0:image_per_class,:,:]), axis=0)
# else:
#   dataset_mnist = np.concatenate((dataset[y_data == 0][0:image_per_class,:,:], \
#                                   dataset[y_data == 1][0:image_per_class,:,:]), axis=0)
# x_train = dataset_mnist.astype("uint8")

# image_per_class = 400
dataset = np.zeros((x_data.shape[0],x_data.shape[1]//resize_ratio,x_data.shape[2]//resize_ratio))
for i in range(x_data.shape[0]):
    dataset[i,:,:] = cv2.resize(x_data[i,:,:], (x_data.shape[1]//resize_ratio,x_data.shape[2]//resize_ratio), interpolation = cv2.INTER_AREA)
dataset_mnist = np.concatenate(([dataset[y_data == k][0:image_per_class,:,:] for k in range(number_of_classes)]), axis=0)

dataset_mnist_y = np.concatenate(([y_data[y_data == k][0:image_per_class,] for k in range(number_of_classes)]), axis=0)

dataset_mnist_test = np.concatenate(([dataset[y_data == k][-1000:,:,:] for k in range(number_of_classes)]), axis=0)
dataset_mnist_test_y = np.concatenate(([y_data[y_data == k][-1000:,] for k in range(number_of_classes)]), axis=0)


# # Data preparation back up
# x_train = dataset_mnist.astype("uint8")
# x_train = np.expand_dims(x_train, 1)
# tensor_x = torch.Tensor(x_train)

# # dataset_mnist_y = dataset_mnist_y.astype("uint8")
# y_train = dataset_mnist_y
# tensor_y = torch.Tensor(y_train).type(torch.LongTensor)
# my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
# train_data = DataLoader(my_dataset, batch_size=20) # create your dataloader

# x_test = dataset_mnist_test.astype("uint8")
# x_test = np.expand_dims(x_test, 1)
# tensor_x_test = torch.Tensor(x_test)

# # dataset_mnist_y = dataset_mnist_y.astype("uint8")
# y_test = dataset_mnist_test_y
# tensor_y_test = torch.Tensor(y_test).type(torch.LongTensor)
# my_dataset_test = TensorDataset(tensor_x_test,tensor_y_test) # create your datset
# test_data = DataLoader(my_dataset_test, batch_size=2000) # create your dataloader


################################
# Data preparation
unlabeled_pool_x = dataset_mnist.astype("uint8")
unlabeled_pool_x = np.expand_dims(unlabeled_pool_x, 1)

# dataset_mnist_y = dataset_mnist_y.astype("uint8")
unlabeled_pool_y = dataset_mnist_y

# initial dataset
print("SHAPE ", unlabeled_pool_x.shape[0])
initial_size = number_of_classes*10
# Choose first 30 samples (10 from each class) randomly
initial_indices = list()
for i in range(number_of_classes):
    initial_indices = initial_indices + list(np.random.choice(unlabeled_pool_x.shape[0]//number_of_classes, int(initial_size/number_of_classes), replace=False) + i*unlabeled_pool_x.shape[0]//number_of_classes)
    # print(initial_indices_0)

# initial_indices_0 = np.random.choice(int(unlabeled_pool_x.shape[0]/2), int(initial_size/2), replace=False)
# print(initial_indices_0)

# initial_indices_1 = np.random.choice(int(unlabeled_pool_x.shape[0]/2), int(initial_size/2), replace=False) + int(unlabeled_pool_x.shape[0]/2)
# # print(type(initial_indices_1))
# initial_indices = list(initial_indices_0) + list(initial_indices_1)


print(initial_indices)
# Add newly selected samples to the labeled dataset
x_train = unlabeled_pool_x[initial_indices]
y_train = unlabeled_pool_y[initial_indices]
print(unlabeled_pool_x.shape)
# Remove newly selected samples from the unlabeled dataset
unlabeled_pool_x = np.delete(unlabeled_pool_x, initial_indices, axis=0)
unlabeled_pool_y = np.delete(unlabeled_pool_y, initial_indices, axis=0)
print(x_train.shape)
print(unlabeled_pool_y.shape)

tensor_x = torch.Tensor(x_train)
tensor_y = torch.Tensor(y_train).type(torch.LongTensor)
dataset = TensorDataset(tensor_x,tensor_y) # create your datset
train_data = DataLoader(dataset, batch_size=20) # create your dataloader

x_test = dataset_mnist_test.astype("uint8")
x_test = np.expand_dims(x_test, 1)
tensor_x_test = torch.Tensor(x_test)

# dataset_mnist_y = dataset_mnist_y.astype("uint8")
y_test = dataset_mnist_test_y
tensor_y_test = torch.Tensor(y_test).type(torch.LongTensor)
dataset_test = TensorDataset(tensor_x_test,tensor_y_test) # create your datset
test_data = DataLoader(dataset_test, batch_size=2000) # create your dataloader

################################

# print(x_train.shape)
# print(y_train.shape)
# print(train_data)
net, test_acc = train_model(net,train_data,test_data,epoch)


active_learning_cycle = 30
# images_per_cycle = 20
num_of_selection = 10
test_accs = []
for cycle in range(active_learning_cycle):
    print('Active learning cycle' , cycle+1 , " / ", active_learning_cycle)
    outputs = net(torch.Tensor(unlabeled_pool_x)).detach().cpu().numpy()
    # print("out ", outputs)
    # print()
    # selected_samples = choose_sample(fake_outputs,num_of_classes,'entropy',num_of_selection)
    selected_samples = choose_sample(outputs,number_of_classes,'least confident',num_of_selection)
    # selected_samples = choose_sample_distances(outputs,number_of_classes,'least confident',num_of_selection,distances)
    
    x_train_selected = unlabeled_pool_x[selected_samples]
    y_train_selected = unlabeled_pool_y[selected_samples]
    print(unlabeled_pool_x.shape)
    unlabeled_pool_x = np.delete(unlabeled_pool_x, selected_samples, axis=0)
    unlabeled_pool_y = np.delete(unlabeled_pool_y, selected_samples, axis=0)
    # print(x_train_selected.shape)
    # print(x_train.shape)
    x_train = np.append(x_train, x_train_selected, axis=0)
    y_train = np.append(y_train, y_train_selected, axis=0)
    print(y_train.shape)
    
    tensor_x = torch.Tensor(x_train)
    tensor_y = torch.Tensor(y_train).type(torch.LongTensor)
    dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    train_data = DataLoader(dataset, batch_size=20) # create your dataloader

    # net = Net()    
    # net.load_state_dict(torch.load('learner.pth'))
    net = Net()

    # make_dot(y.mean(), params=dict(net.named_parameters()))
    # torch.save(net.state_dict(), 'learner.pth')
    # [66.16666666666667, 65.5, 33.333333333333336, 64.4, 66.46666666666667, 33.333333333333336, 65.53333333333333, 33.36666666666667, 33.333333333333336, 97.7]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net, test_acc = train_model(net,train_data,test_data,epoch)
    test_accs.append(test_acc)
print(test_accs)
exp = 2
# np.save("results/" + str(exp)+"_least_confidence_distance.npy", np.array(test_accs))
np.save("results/" + str(exp)+"_least_confidence.npy", np.array(test_accs))