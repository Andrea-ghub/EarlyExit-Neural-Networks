import torch, torchvision
import pickle
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms


class EarlyExitCustomNet64(nn.Module):
    def __init__(self, device, resnext = True, T=0, weights = None):
        super(EarlyExitCustomNet64, self).__init__()

        self.T = T
        if resnext:
          self.model = torchvision.models.resnext50_32x4d(weights=weights)
        else: self.model = torchvision.models.resnet50(weights=weights)
        self.model.fc = nn.Linear(in_features = 2048, out_features = 200, bias = True)
        self.blocks = list(self.model.children())
        self.lung = len(self.blocks)
        self.frac = self.lung//3   #3 exits
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.device = device

        self.f1 = nn.Sequential(*self.blocks[0:self.frac+2])
        self.f2 = nn.Sequential(*self.blocks[self.frac+2:2*self.frac+1])
        self.f3 = nn.Sequential(*self.blocks[2*self.frac+1:-1])


        self.exit1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Flatten(),
            # nn.Linear(in_features = 4*16384, out_features = 200, bias = True)
            nn.Linear(in_features = 16384, out_features = 200, bias = True)
        )

        self.exit2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Flatten(),
            # nn.Linear(in_features = 4*4096, out_features = 200, bias = True)
            nn.Linear(in_features = 4096, out_features = 200, bias = True)
        )

        self.exit3 = nn.Sequential(
            self.f3,
            nn.Flatten(),
            nn.Linear(in_features = 2048, out_features = 200, bias = True)
        )


    def entropy_mask(self, x, T):

        x_soft = self.softmax(x)
        entropy = torch.sum(torch.special.entr(x_soft), dim = 1)
        mask = entropy < T

        return mask


    def exit_predictions(self, x, indices, predictions, T):


            # TAKE THE EXIT DATA INDICES
            mask = self.entropy_mask(x,T)
            out_indices = indices[mask]


            # COMPUTE THE PREDICTIONS
            _, out = torch.max(x[mask], 1)
            predictions[out_indices] = out


            # UPDATE THE VARIABLES
            not_mask = torch.bitwise_not(mask)
            indices = indices[not_mask]

            return not_mask, indices, predictions


    def forward(self, x):
        T = self.T
        # print("FORWARD CALL")
        if self.training:
          x = self.f1(x)
          y = self.exit1(x)
          x = self.f2(x)
          z = self.exit2(x)
          x = self.exit3(x)

          out = [y,z,x]
          return out

        else:
          predictions = torch.zeros(x.shape[0], dtype = torch.int64, device=self.device)
          indices = torch.arange(0, x.shape[0], device=self.device)


          x = self.f1(x)
          x_1 = self.exit1(x)

          mask, indices, predictions = self.exit_predictions(x_1, indices, predictions, T)

          x = x[mask]
          x = self.f2(x)
          x_2 = self.exit2(x)

          mask, indices, predictions = self.exit_predictions(x_2, indices, predictions, T)

          x = x[mask]
          x_3 = self.exit3(x)

          _, out = torch.max(x_3, 1)
          predictions[indices] = out


          return predictions


class EarlyExitCustomNet128(nn.Module):
    def __init__(self, device, resnext = True, T = 0, weights = None):
        super(EarlyExitCustomNet128, self).__init__()

        self.T = T
        if resnext:
          self.model = torchvision.models.resnext50_32x4d(weights=weights)
        else: self.model = torchvision.models.resnet50(weights=weights)
        
        self.model.fc = nn.Linear(in_features = 2048, out_features = 200, bias = True)
        self.blocks = list(self.model.children())
        self.lung = len(self.blocks)
        self.frac = self.lung//3   #3 exits
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.device = device

        self.f1 = nn.Sequential(*self.blocks[0:self.frac+2])
        self.f2 = nn.Sequential(*self.blocks[self.frac+2:2*self.frac+1])
        self.f3 = nn.Sequential(*self.blocks[2*self.frac+1:-1])


        self.exit1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(in_features = 4*16384, out_features = 200, bias = True)
            # nn.Linear(in_features = 16384, out_features = 200, bias = True)
        )

        self.exit2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(in_features = 4*4096, out_features = 200, bias = True)
            # nn.Linear(in_features = 4096, out_features = 200, bias = True)
        )

        self.exit3 = nn.Sequential(
            self.f3,
            nn.Flatten(),
            # self.blocks[-1]
            nn.Linear(in_features = 2048, out_features = 200, bias = True)
        )


    def entropy_mask(self, x, T):

        x_soft = self.softmax(x)
        entropy = torch.sum(torch.special.entr(x_soft), dim = 1)
        mask = entropy < T

        return mask


    def exit_predictions(self, x, indices, predictions, T):


            # TAKE THE EXIT DATA INDICES
            mask = self.entropy_mask(x,T)
            out_indices = indices[mask]


            # COMPUTE THE PREDICTIONS
            _, out = torch.max(x[mask], 1)
            predictions[out_indices] = out


            # UPDATE THE VARIABLES
            not_mask = torch.bitwise_not(mask)
            indices = indices[not_mask]

            return not_mask, indices, predictions


    def forward(self, x):
        T = self.T
        if self.training:
          x = self.f1(x)
          y = self.exit1(x)
          x = self.f2(x)
          z = self.exit2(x)
          x = self.exit3(x)

          out = [y,z,x]
          return out

        else:
          predictions = torch.zeros(x.shape[0], dtype = torch.int64, device=self.device)
          indices = torch.arange(0, x.shape[0], device=self.device)


          x = self.f1(x)
          x_1 = self.exit1(x)

          mask, indices, predictions = self.exit_predictions(x_1, indices, predictions, T)

          x = x[mask]
          x = self.f2(x)
          x_2 = self.exit2(x)

          mask, indices, predictions = self.exit_predictions(x_2, indices, predictions, T)

          x = x[mask]
          x_3 = self.exit3(x)

          _, out = torch.max(x_3, 1)
          predictions[indices] = out

          return predictions



def rotate_90(image):
    return transforms.functional.rotate(image,90)

def rotate_180(image):
    return transforms.functional.rotate(image,180)

def rotate_270(image):
    return transforms.functional.rotate(image,270)


class CreateDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path, augmentation=None, train = True, resize = 64):
        """
        Arguments:
            dataset (string): Path to the dataset file.
            image_size (int): size to resize the image
        """

        self.augmentation = augmentation
        self.path = data_path
        self.name = "train_dataset_raw.pkl"
        self.train = train
        self.resize = resize

        if not self.train:
            self.name = "validation_dataset.pkl"

            self.tfs = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((self.resize,self.resize),interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        else:

            self.tfs = transforms.Compose([

            transforms.ToPILImage(),
            transforms.Resize((self.resize,self.resize),interpolation=transforms.InterpolationMode.BICUBIC),

            # transforms.Resize((72, 72)),

            # transforms.v2.RandomCrop(64),  #new added

            # transforms.v2.RandomApply([

            #     transforms.v2.Lambda(rotate_90)
            # ],p=0.25),

            # transforms.v2.RandomApply([

            #     transforms.v2.Lambda(rotate_180)
            # ],p=0.25),

            # transforms.v2.RandomApply([

            #     transforms.v2.Lambda(rotate_270)
            # ],p=0.25),


            transforms.v2.RandomHorizontalFlip(0.30),
            transforms.v2.RandomVerticalFlip(0.30),


#             transforms.v2.RandomApply([
#                 transforms.v2.RandomChannelPermutation()
#             ],p=0.25),



            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])



        with open(self.path + self.name, 'rb') as f:
             self.dataset = pickle.load(f)

        with open(self.path + 'labels_map.pkl', 'rb') as f:
             self.labels_map = pickle.load(f)

#         self.tfs = transforms.Compose([

#             transforms.ToPILImage(),
#             transforms.v2.RandomHorizontalFlip(),
#             transforms.v2.RandomVerticalFlip(),
#             transforms.v2.RandomCrop(54),
#             transforms.v2.RandomAffine(degrees=0,translate=(0,0.1),shear=(0,0.1)),
#             transforms.v2.RandomRotation([-15,15]),
#             transforms.v2.Pad(random.randint(0,5),0),
#             transforms.v2.ColorJitter(brightness=0.3,contrast=0.3),
#             transforms.v2.Resize(64),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])



    def __len__(self):
        return len(self.dataset["labels"])


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        images = self.dataset["images"][idx]
        labels = self.dataset["labels"][idx]
        labels = torch.LongTensor([self.labels_map[label.item()] for label in labels])



        if self.augmentation:
            images = self.tfs(images)

        sample = {'images': images, 'labels': labels}

        return sample