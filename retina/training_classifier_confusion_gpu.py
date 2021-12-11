import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import models
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
# from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix
import pickle
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
folder_path = os.path.dirname(os.path.abspath(__file__))
torch.cuda.empty_cache()
import gc
gc.collect()
from sklearn.metrics import roc_auc_score

number_of_classes = 2
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
        self.conv3 = nn.Conv2d(20, 30, 5)
        hidden_count = 30
        self.fc1   = nn.Linear(30*24*24, hidden_count)
        self.fc2   = nn.Linear(hidden_count, hidden_count)
        self.fc3   = nn.Linear(hidden_count, number_of_classes)
        # self.fc3   = nn.Linear(84, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        # x = self.pool(x)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 30*24*24)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, 5)
#         self.pool  = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(10, 20, 5)
#         hidden_count = 30
#         self.fc1   = nn.Linear(20*53*53, hidden_count)
#         self.fc2   = nn.Linear(hidden_count, number_of_classes)
#         # self.fc3   = nn.Linear(84, 2)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         # print(x.shape)
#         # x = self.pool(x)
#         # x = F.relu(self.conv1(x))
#         # x = F.relu(self.conv2(x))
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         # print("try ", x.shape)
#         x = x.view(-1, 20*53*53)
#         x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         x = self.fc2(x)
#         x = self.softmax(x)
#         return x

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
            # print(outputs)
            # print(labels)
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
    # cm = confusion_matrix(cm_labels, cm_predicted)    
    
    # auc = roc_auc_score(labels.to("cpu"), predicted.to("cpu"))
    # print('AUC of the network on the 4000 test images: %f %%' % (auc))
    
    print('Accuracy of the network on the 4000 test images: %f %%' % (100 * correct / total))
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
    # print(class_total)
    for i in range(number_of_classes):
        print('Accuracy of %5s : %4f %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    # print(labels)    
    
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

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6,5)) 

    
    sns.set(font_scale = 2)
    sns.heatmap(df_confusion, annot=True, annot_kws={"size": 20}, ax=ax, fmt='g')
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.title(title)
    plt.savefig("plots/"+title+".png")



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
distances = load_distances_matrix("distances_retina.npy")
# distances = np.load(ospath.join(folder_path, "distances.npy"))

exp=3
for e in range(exp):
    j=0
# for j in range(0,5):
    # print("I ALPHA ", i, j*0.25)
    alpha=j*0.25
    seed=e
    torch.manual_seed(seed)
    np.random.seed(seed)
    #net = Net()
    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc =  nn.Sequential(
                nn.Linear(num_ftrs, 2),
                nn.Sigmoid())
    
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
       
    # Load data
    folder_path = ospath.dirname(ospath.abspath(__file__))
    partition_file_6000 = ospath.join(folder_path,'rop_data/6000Partitions.p')
    partition_file_6000 = pickle.load(open(partition_file_6000, 'rb'), encoding='latin1')

    img_folder_6000=ospath.join(folder_path,'rop_data/preprocessed_JamesCode/')
    
    part_rsd_test = partition_file_6000['RSDTestPartition']
    img_names = partition_file_6000['imgNames']
    rsd_labels_plus = partition_file_6000['RSDLabelsPlus']
    rsd_labels_prep = partition_file_6000['RSDLabelsPreP']
    # test on all 5000
    ind_rsd_test = part_rsd_test[0].astype(np.int)
    for k in [1, 2, 3, 4]:
        ind_rsd_test = np.append(ind_rsd_test, part_rsd_test[k].astype(np.int))
    abs_thr='pre-plus'
    if abs_thr == 'plus':
        abs_labels = rsd_labels_plus[ind_rsd_test]
    else:
        abs_labels = rsd_labels_prep[ind_rsd_test]
        
    print(np.where(abs_labels<=0)[0].shape)
    print(np.where(abs_labels>0)[0].shape)
    
    img_test_list = [img_folder_6000 + img_names[int(order)] + '.png' for order in ind_rsd_test]
    img_test_list = np.asarray(img_test_list)
    number_of_fewer = np.where(abs_labels>0)[0].shape[0]
    img_test_list =np.concatenate((img_test_list[np.where(abs_labels<=0)[0][0:number_of_fewer]] , img_test_list[np.where(abs_labels>0)[0]]))                              
    retina_images_list = img_test_list#[name.split("test/")[1] for name in img_test_list]
    labels =np.concatenate((abs_labels[np.where(abs_labels<=0)[0][0:number_of_fewer]] , abs_labels[np.where(abs_labels>0)[0]]))                              
    resize_ratio = 1
    abs_labels = labels
    abs_labels[np.where(abs_labels<0)]=0

    
    image = cv2.imread(retina_images_list[0])
    image = cv2.resize(image, (image.shape[0]//resize_ratio, image.shape[1]//resize_ratio), interpolation = cv2.INTER_AREA)
    dataset = np.zeros((len(retina_images_list), image.shape[0], image.shape[1], 3))  
    
    print(len(img_test_list),'shape')
    dataset_size = 1968
    for k in range(dataset_size):          
      image = cv2.imread(retina_images_list[k])
      image = cv2.resize(image, (image.shape[0]//resize_ratio, image.shape[1]//resize_ratio), interpolation = cv2.INTER_AREA)
      # image = np.transpose(image,(2,0,1))
      dataset[k] = image

    indices = np.arange(dataset_size).reshape(2,-1).flatten("F")
    test_indices = indices[-dataset_size//5:]
    train_indices = indices[:-dataset_size//5]
    print(retina_images_list[train_indices])
    
    train_indices = train_indices[0:1000]
    dataset_mnist_test = dataset[test_indices]
    dataset_mnist_test_y = abs_labels[test_indices]
    dataset_mnist  = dataset[train_indices]
    dataset_mnist_y = abs_labels[train_indices]
    
    
    classes = ([str(i) for i in np.unique(dataset_mnist_test_y)])
 
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
    x_train = np.transpose(unlabeled_pool_x[initial_indices][:,0,:,:,:], (0,3,1,2))
                 
    y_train = unlabeled_pool_y[initial_indices]
    print(unlabeled_pool_x.shape)
    # Remove newly selected samples from the unlabeled dataset
    unlabeled_pool_x = np.delete(unlabeled_pool_x, initial_indices, axis=0)
    unlabeled_pool_y = np.delete(unlabeled_pool_y, initial_indices, axis=0)
    unlabeled_pool_x = np.transpose(unlabeled_pool_x[:,0,:,:,:], (0,3,1,2))
    print(x_train.shape)
    print(unlabeled_pool_y.shape)
    
    tensor_x = torch.Tensor(x_train).to(device)
    tensor_y = torch.Tensor(y_train).type(torch.LongTensor).to(device)
    dataset = TensorDataset(tensor_x,tensor_y) # create your dataset
    batch_size = 350
    train_data = DataLoader(dataset, batch_size=batch_size) # create your dataloader
    
    x_test = dataset_mnist_test.astype("uint8")
    x_test = np.expand_dims(x_test, 1)
    tensor_x_test = torch.Tensor(np.transpose(x_test[:,0,:,:,:], (0,3,1,2))).to(device)
    
    y_test = dataset_mnist_test_y

    tensor_y_test = torch.Tensor(y_test).type(torch.LongTensor).to(device)
    
    dataset_test = TensorDataset(tensor_x_test,tensor_y_test) # create your dataset
    
    test_data = DataLoader(dataset_test, batch_size=batch_size ) # create your dataloader
    
    ################################

    epoch=1000
    net, test_acc = train_model(net,train_data,test_data,epoch)

    active_learning_cycle = 15

    num_of_selection = 10

    test_accs = [test_acc]
    # test_aucs= [auc]
    batch = 5
    for cycle in range(active_learning_cycle):
        unlabeled_pool_x = torch.Tensor(unlabeled_pool_x).type(torch.FloatTensor).to(device)
        print('Active learning cycle' , cycle+1 , " / ", active_learning_cycle)
        unlabeled_pool_x_dataloader = DataLoader(unlabeled_pool_x, batch_size=batch_size )
        outputs = np.zeros((unlabeled_pool_x.shape[0], 2))
        i = 0
        for data in unlabeled_pool_x_dataloader:
            inputs = data
            output = net(data) 
            outputs[i:i+len(data),:] = output.cpu().detach().numpy()
            i += len(data)
        
        # outputs = torch.cat([net(data) for data in unlabeled_pool_x_dataloader])
        # outputs = torch.cat([net(torch.Tensor(unlabeled_pool_x[i*batch:min(batch+i*batch, unlabeled_pool_x.shape[0])]).to(device)) for i in range(unlabeled_pool_x.shape[0]//batch)])
        # net(unlabeled_pool_x)
        # selected_samples = choose_sample(outputs,number_of_classes,'entropy',num_of_selection)
        # selected_samples = choose_sample(outputs,number_of_classes,'least confident',num_of_selection)
        
        # selected_samples = choose_sample_distances(outputs,number_of_classes,'least confident',num_of_selection,distances,alpha)
        
        # selected_samples = choose_sample_distances(outputs,number_of_classes,'entropy',num_of_selection,distances,alpha)
        
        # selected_samples = choose_sample_distances(outputs,number_of_classes,'margin sampling',num_of_selection,distances,alpha)
        
        selected_samples = choose_sample_distances(outputs,number_of_classes,'random',num_of_selection,distances,alpha)
        
        unlabeled_pool_x = unlabeled_pool_x.cpu()
        x_train_selected = unlabeled_pool_x[selected_samples]
        y_train_selected = unlabeled_pool_y[selected_samples]
        print(unlabeled_pool_x.shape)
        unlabeled_pool_x = np.delete(unlabeled_pool_x, selected_samples, axis=0)
        unlabeled_pool_y = np.delete(unlabeled_pool_y, selected_samples, axis=0)

        x_train = np.append(x_train, x_train_selected, axis=0)
        y_train = np.append(y_train, y_train_selected, axis=0)
        print(y_train.shape)
        del outputs
        tensor_x = torch.Tensor(x_train).to(device)
        tensor_y = torch.Tensor(y_train).type(torch.LongTensor).to(device)
        dataset = TensorDataset(tensor_x,tensor_y) # create your datset
        train_data = DataLoader(dataset, batch_size=batch_size ) # create your dataloader
        
        # net = Net()
    
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
        
        net.train()
        
        net, test_acc = train_model(net,train_data,test_data,epoch)
        del tensor_x,tensor_y
        test_accs.append(test_acc)
        # test_aucs.append(auc)
        
    # plot_confusion_matrix(cm, "Least Confidence-Confusion Matrix for exp="+str(e)+" alpha="+str(alpha))#+" cycle="+str(cycle))   
    # print(test_accs)
    print(test_accs)
    # np.save("results/2class/" + str(seed)+"_least_confidence_distance_"+str(alpha)+".npy", np.array(test_accs))
    # np.save("results/2class/" + str(seed)+"_entropy_distance_"+str(alpha)+".npy", np.array(test_accs))
    # np.save("results/2class/" + str(seed)+"_margin_sampling_distance_"+str(alpha)+".npy", np.array(test_accs))
    np.save("results/2class/" + str(seed)+"_random_"+str(alpha)+".npy", np.array(test_accs))
