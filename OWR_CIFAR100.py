import torch
import torchvision
from torchvision.datasets import CIFAR100
from torchvision.datasets import VisionDataset
from torchvision.transforms import transforms
import torchvision.utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

NUM_CLASSES = 100
BATCH_SIZE = 128
DATA_DIR = './CIFAR_100'  
N_GROUPS_FOR_TRAINING = 10   # Numero di gruppi di classi da usare in fase di training (1: usa solo il primo gruppo, 10: usa tutti i gruppi di classi)


class DatasetCifar100(VisionDataset):
    def __init__(self, split, closedWorld = True, rand_seed=None):
        super(DatasetCifar100, self).__init__(root=0)

        means, stds = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)              # Normalizes tensor with mean and standard deviation of CIFAR 100
        if split=="train":
            self.transform = transforms.Compose([
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.ToTensor(),            # Turn PIL Image to torch.Tensor
                                                    transforms.Normalize(mean=means, std=stds)          # Normalizes tensor with mean and standard deviation of CIFAR 100
                                                ])
        else:
            self.transform= transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=means, std=stds)               # Normalizes tensor with mean and standard deviation of CIFAR 100                                                                                                
                                            ])
            

        #Decide the order of the classes
        if rand_seed == None:
          self.classes=[36, 61, 49, 58, 92, 90, 68, 32, 28, 52,\
                        47, 87, 1, 41, 93, 6, 88, 12, 38, 91,\
                        81, 33, 8, 48, 60, 27, 50, 17, 56, 97,\
                        34, 42, 84, 66, 62, 26, 29, 51, 3, 72,\
                        39, 9, 37, 85, 13, 25, 11, 67, 99, 74,\
                        30, 2, 64, 71, 19, 35, 31, 63, 54, 15,\
                        43, 73, 40, 55, 7, 78, 14, 10, 70, 44,\
                        0, 86, 79, 57, 75, 46, 83, 82, 22, 4,\
                        45, 18, 89, 5, 59, 21, 95, 96, 69, 16,\
                        98, 23, 80, 65, 76, 77, 20, 24, 94, 53]
         
        else:
          np.random.seed(rand_seed)
          self.classes = np.array([i for i in range(NUM_CLASSES)])
          np.random.shuffle(self.classes)

        if split=="train":
            self.dataset = torchvision.datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=self.transform)
        else:
            self.dataset = torchvision.datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=self.transform) # cosa cambia qui? train:true/false

        self.dict_class_label = {c:i for i, c in enumerate(self.classes)} 

        self.classOne = self.classes[0:50]
        self.classTwo = self.classes[50:100]

        
        #Otteniamo un dizionario {classe: indice} = {56:0}, {99,1},..
        #Le classi in classes sono disordinate,
        #label è l'indice con cui arriva quella classe e sarà l'indice(neurone di output) della predizione(la rete mi dice che quella classe sta a 0 e non che è 56,
        #quindi devo fare mappping!!!)

      
        #if split=="train":
        #    self.dataset.targets = [self.dict_class_label[target] for target in self.dataset.targets] #mappo i target che arrivano in base all'ordine delle classi scelto
         #   self.CLASSES = self.dataset.classes #insieme delle classi (100) a parole
          #  self.dataset.class_to_idx = {self.CLASSES[i]:cl for i,cl in enumerate(self.classes)}
       # else:
        #    self.dataset.targets = [self.dict_class_label[target] for target in self.dataset.targets] 
         #   self.CLASSES = self.dataset.classes #insieme delle classi (100) a parole
          #  self.dataset.class_to_idx = {self.CLASSES[i]:cl for i,cl in enumerate(self.classes)}
            #Mapping classe-numero classe nell'ordine di arrivo es. apple (classe 0): 33
           
    
    #def __getitem__(self, index):
    #    return self.dataset[index][0], self.dataset[index][1]

    def __getitem__(self, index):
        return index, self.dataset[index][0], self.dataset[index][1]

    def get_classes(self):
      return self.dataset.class_to_idx
    
    def get_targets(self):
      return self.dataset.targets
    
    def get_transform(self):
      return self.transform

    #Funzione che salva gli indici del vettore targets (che contiene le classi corrispondenti alle 10 in questione)
    def get_indexes_from_labels(self, labels):
        targets = self.dataset.targets #sono tutte le etichette dei dati in arrivo già mappate secondo l'ordine casuale delle classi scelto 
        indexes = []
        for i,target in enumerate(targets):
          if target in labels:
              indexes.append(i)
        return indexes #serve a trovare gli indici delle immagini appartenenti a quelle 10 classi che mi interessano
    
    def train_validation_split(self, indexes, train_size=0.9, random_state=None):
        # split indexes between train and validation
        targets = self.dataset.targets
        train_indexes, val_indexes, _, _ = train_test_split(list(range(0, len(indexes))), list(range(0, len(indexes))), train_size=train_size, stratify=[targets[i] for i in indexes], random_state=random_state)
        np_indexes = np.array(indexes)
        return list(np_indexes[train_indexes]), list(np_indexes[val_indexes])

        
    
