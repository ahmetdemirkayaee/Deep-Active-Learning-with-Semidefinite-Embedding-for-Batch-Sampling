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

from sklearn.metrics import confusion_matrix

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
    # epochs=1000
    epochs_no_change = 0
    prev_loss = 0
    max_epoch_no_change = 5
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
        # print(abs(running_loss-prev_loss ))
        if abs(running_loss-prev_loss )<0.00001:
            epochs_no_change+=1
        prev_loss = running_loss
        if epochs_no_change==max_epoch_no_change:
            print("Epoch: ", epoch)
            break
    
    print('Finished Training')
    

    correct = 0
    total = 0
    cm_labels = []
    cm_predicted = []
    with torch.no_grad():
        for data in testloader:
            # print(data)
            images, labels = data
            # print(type(labels))
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            cm_labels.append(labels)
            cm_predicted.append(predicted)
        cm_labels = torch.cat(cm_labels, dim=0)
        cm_predicted = torch.cat(cm_predicted, dim=0)
    # print(labels)
    # print(predicted)
    cm = confusion_matrix(cm_labels, cm_predicted)    
    
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
    # print(labels)    
    
    return net, test_acc, cm

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

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    import seaborn as sns
    plt.figure()
    sns.heatmap(df_confusion, annot=True, annot_kws={"size": 20}, fmt='g')
    # plt.matshow(df_confusion)#, cmap=cmap) # imshow
    plt.title(title)
    # plt.colorbar()
    # tick_marks = np.arange(len(df_confusion[0]))
    plt.savefig("confusion_matrices/3class/"+title+".png")
    # # plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    # # plt.yticks(tick_marks, df_confusion.index)
    # #plt.tight_layout()
    # # plt.ylabel(df_confusion.index.name)
    # # plt.xlabel(df_confusion.columns.name)


classes = ('0', '1', '2')


# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_distances_matrix(file_path):
    distances = np.load(ospath.join(folder_path, file_path))
    return distances

# Load distances matrix
distances = load_distances_matrix("distances_3.npy")
# distances = np.load(ospath.join(folder_path, "distances.npy"))

exp=3
for e in range(exp):
    for j in range(0,5):
        # print("I ALPHA ", i, j*0.25)
        alpha=j*0.25
        seed=e
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = Net()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
        
        folder_path = os.path.dirname(os.path.abspath(__file__))
        x_data = np.load(ospath.join(folder_path, "mnist_train_x.npy"))
        y_data = np.load(ospath.join(folder_path, "mnist_train_y.npy"))
        
        resize_ratio = 1
        image_per_class = 400

        # image_per_class = 400
        dataset = np.zeros((x_data.shape[0],x_data.shape[1]//resize_ratio,x_data.shape[2]//resize_ratio))
        for i in range(x_data.shape[0]):
            dataset[i,:,:] = cv2.resize(x_data[i,:,:], (x_data.shape[1]//resize_ratio,x_data.shape[2]//resize_ratio), interpolation = cv2.INTER_AREA)
        dataset_mnist = np.concatenate(([dataset[y_data == k][0:image_per_class,:,:] for k in range(number_of_classes)]), axis=0)
        
        dataset_mnist_y = np.concatenate(([y_data[y_data == k][0:image_per_class,] for k in range(number_of_classes)]), axis=0)
        
        dataset_mnist_test = np.concatenate(([dataset[y_data == k][-1000:,:,:] for k in range(number_of_classes)]), axis=0)
        dataset_mnist_test_y = np.concatenate(([y_data[y_data == k][-1000:,] for k in range(number_of_classes)]), axis=0)
                
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
        dataset = TensorDataset(tensor_x,tensor_y) # create your dataset
        train_data = DataLoader(dataset, batch_size=20) # create your dataloader
        
        x_test = dataset_mnist_test.astype("uint8")
        x_test = np.expand_dims(x_test, 1)
        tensor_x_test = torch.Tensor(x_test)
        
        # dataset_mnist_y = dataset_mnist_y.astype("uint8")
        y_test = dataset_mnist_test_y
        tensor_y_test = torch.Tensor(y_test).type(torch.LongTensor)
        
        dataset_test = TensorDataset(tensor_x_test,tensor_y_test) # create your dataset
        
        test_data = DataLoader(dataset_test, batch_size=2000) # create your dataloader
        
        ################################
        
        epoch=1000
        net, test_acc, cm = train_model(net,train_data,test_data,epoch)
        # outputs = net(torch.Tensor(unlabeled_pool_x)).detach().cpu().numpy()
        
        active_learning_cycle = 15
        # images_per_cycle = 20
        num_of_selection = 10
        # alpha = 0.25
        test_accs = [test_acc]
        for cycle in range(active_learning_cycle):
            print('Active learning cycle' , cycle+1 , " / ", active_learning_cycle)
            outputs = net(torch.Tensor(unlabeled_pool_x)).detach().cpu().numpy()

            # selected_samples = choose_sample(outputs,number_of_classes,'entropy',num_of_selection)
            # selected_samples = choose_sample(outputs,number_of_classes,'least confident',num_of_selection)
            
            # selected_samples = choose_sample_distances(outputs,number_of_classes,'least confident',num_of_selection,distances,alpha)
            
            selected_samples = choose_sample_distances(outputs,number_of_classes,'entropy',num_of_selection,distances,alpha)
            
            # selected_samples = choose_sample_distances(outputs,number_of_classes,'margin sampling',num_of_selection,distances,alpha)
            
            # selected_samples = choose_sample_distances(outputs,number_of_classes,'random',num_of_selection,distances,alpha)
            
            x_train_selected = unlabeled_pool_x[selected_samples]
            y_train_selected = unlabeled_pool_y[selected_samples]
            print(unlabeled_pool_x.shape)
            unlabeled_pool_x = np.delete(unlabeled_pool_x, selected_samples, axis=0)
            print(unlabeled_pool_x.shape)
            unlabeled_pool_y = np.delete(unlabeled_pool_y, selected_samples, axis=0)

            x_train = np.append(x_train, x_train_selected, axis=0)
            y_train = np.append(y_train, y_train_selected, axis=0)
            print(y_train.shape)
            
            tensor_x = torch.Tensor(x_train)
            tensor_y = torch.Tensor(y_train).type(torch.LongTensor)
            dataset = TensorDataset(tensor_x,tensor_y) # create your datset
            train_data = DataLoader(dataset, batch_size=20) # create your dataloader
            
            # net = Net()
        
            # criterion = nn.CrossEntropyLoss()
            # optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
            
            net.train()
            
            net, test_acc, cm = train_model(net,train_data,test_data,epoch)
            
            test_accs.append(test_acc)
            
        # plot_confusion_matrix(cm, "Confusion Matrix for exp="+str(e)+" alpha="+str(alpha))#+" cycle="+str(cycle))   
        print(test_accs)
        # np.save("results/" + str(exp)+"_least_confidence.npy", np.array(test_accs))
        # np.save("results/3class/" + str(seed)+"_least_confidence_distance_"+str(alpha)+".npy", np.array(test_accs))
        # np.save("results/3class/" + str(seed)+"_margin_sampling_distance_"+str(alpha)+".npy", np.array(test_accs))
        np.save("results/3class/" + str(seed)+"_entropy_distance_"+str(alpha)+".npy", np.array(test_accs))
        # np.save("results/3class/" + str(seed)+"_random_"+str(alpha)+".npy", np.array(test_accs))