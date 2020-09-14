import torch
from PIL import Image,ImageOps
import numpy as np
from torchvision import datasets, transforms

class DataHelper:

    def process_image(self, img_path):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array

        Arguments:
            image_path: Path to image
        Returns:
            img_arr: Numpy image array

        '''    
        # TODO: Process a PIL image for use in a PyTorch model
        #return valid_transforms(Image.open(image)).float()

        im = Image.open(img_path)
        width, height = im.size

        if width<height:
            size = 256, 999999999
        else:
            size = 999999999, 256

        im.thumbnail(size=size)

        width, height = im.size
        l_cord = (width - 224) / 2
        r_cord = (width + 224) / 2
        t_cord = (height - 224) / 2
        b_cord = (height + 224) / 2

        im = im.crop((l_cord, t_cord, r_cord , b_cord))

        # Convert to numpy array
        img_arr = np.array(im)/255

        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_arr = (img_arr - mean) / std

        img_arr = img_arr.transpose(2, 0, 1)

        return img_arr
    
    def get_data_loder(self, data_dir='flowers'):
        ''' Get data loaders

        Arguments:
            data_dir: Data directory
        Returns:
            train_datasets: Train dataset
            test_datasets: Test dataset
            valid_datasets: Validation dataset
            train_dataloader: Train dataloader
            test_dataloader: Test dataloader
            valid_dataloader: Validation dataloader
        '''  
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        
        # Batch size. Small batch size for code validation.
        batch_size = 64

        # Random rotation 30 degrees
        # Crop to 224X224 pixel
        # Random hotizontal flip
        # Random vertical flip
        # Convert image to tensor
        # Normalize with means = [0.485, 0.456, 0.406] & standard deviations = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        # Resize image
        # Crop to 224X224 pixel
        # Convert image to tensor
        # Normalize with means = [0.485, 0.456, 0.406] & standard deviations = [0.229, 0.224, 0.225]
        test_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])                                      

        # Resize image
        # Crop to 224X224 pixel
        # Convert image to tensor
        # Normalize with means = [0.485, 0.456, 0.406] & standard deviations = [0.229, 0.224, 0.225]
        valid_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])


        # DONE: Load the datasets with ImageFolder
        # Train dataset with train_transform
        train_datasets = datasets.ImageFolder(train_dir, transform=train_transform)
        # Test dataset with test_transforms
        test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
        # Validation dataset with valid_transforms
        valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)

        # DONE: Using the image datasets and the trainforms, define the dataloaders
        # Train data loader, batch size 32
        train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
        # Test data loader, batch size 32
        test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)
        # Validate data loader, batch size 32
        valid_dataloader = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)
        
        return train_datasets, test_datasets, valid_datasets, train_dataloader, test_dataloader, valid_dataloader