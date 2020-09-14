import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models
from data_util import DataHelper
import numpy as np

class FlowerClassifier:
    def __init__(self, arch='vgg16', hidden_units=4096, drop_p=0.5, lr=0.0001, class_idx_mapping=None):
        ''' Initialize Flower Classifier
        
        Arguments:
            arch: Model Pretrained Architecture. (Default=vgg16)
            hidden_units: Number of hidden units. (Default=4096)
            drop_p: Drop Percentage. (Default=0.5)
            lr: Learn Rate. (Default=0.0001)
            class_idx_mapping: Class to index mapping

        '''
        super().__init__()
        self.arch = arch
        self.hidden_units = hidden_units
        self.drop_p = drop_p
        self.lr = lr
        
        self.model = None        
        self.classifier = None
        if arch == 'vgg16':
            # Load pretrained model vgg16, pretrained = true
            self.model = models.vgg16(pretrained=True)
            # Create classifier
            self.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                          nn.ReLU(),
                          nn.Dropout(drop_p),
                          nn.Linear(hidden_units, 102),
                          nn.LogSoftmax(dim=1))
        elif arch == 'vgg13':
            # Load pretrained model vgg13, pretrained = true
            self.model = models.vgg13(pretrained=True)
            # Create classifier
            self.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                          nn.ReLU(),
                          nn.Dropout(0.5),
                          nn.Linear(hidden_units, 102),
                          nn.LogSoftmax(dim=1))
        elif arch == 'densenet121':
            # Load pretrained model densenet121, pretrained = true
            self.model = models.densenet121(pretrained=True)
            # Create classifier
            self.classifier = nn.Sequential(nn.Linear(1024, hidden_units),
                          nn.ReLU(),
                          nn.Dropout(drop_p),
                          nn.Linear(hidden_units, 102),
                          nn.LogSoftmax(dim=1))
         
        # Frrezing parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # update model classifier
        self.model.classifier = self.classifier  
        
        self.model.class_idx_mapping = class_idx_mapping
        # Criterian to NLLLoss as Log Softmax is used as O/P activation function
        self.criterion = nn.NLLLoss()
        # Using adam optimizer with learn rate only for classifier parameters
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr)
        
    def train(self, train_dataloader, epochs, valid_dataloader, print_result_every, device):
        ''' Train Trained Model
        
        Arguments:
            train_dataloader: Training data set loader
            epochs: Number of iterations
            valid_dataloader: Validation Data Loader
            print_result_every: Print result after every iteration
            device: Device on which training should run. GPU or CUDA
        
        '''
        # Number of steps or images trained
        self.model.train()
        # change to device
        self.model.to(device)

        train_steps = 0
        print("Starting Training")
        #print("Epochs:", epochs)
        #print("Print Result After Steps:", print_result_every)
        for e in range(epochs):
            # Steps loss
            step_loss = 0
            for img_batch, img_labels in train_dataloader:
                # Increment training step
                train_steps +=1              
                #if(train_steps%5 ==0):
                #   print("Train Step:", train_steps, (train_steps % print_result_every))

                # Move image tensor to GPU if available
                img_batch = img_batch.to(device)

                # Move labels tensor to GPU if available
                img_labels = img_labels.to(device)

                # Reset optimizer gradient
                self.optimizer.zero_grad()

                # Get probability log from model
                prob_log = self.model(img_batch)

                # Calculate loss
                loss = self.criterion(prob_log, img_labels)

                # Calculate gradient
                loss.backward()

                # Optimize weights
                self.optimizer.step()

                #Add step or epoch loss
                step_loss += loss.item()

                if (train_steps % print_result_every) == 0:
                    #print("Running validation")
                    validation_loss = 0
                    model_accuracy = 0

                    # Turn off gradients for validation
                    with torch.no_grad():
                        # Set model to evaliation
                        self.model.eval()
                        for imag_vld, labl_vld in valid_dataloader:
                            # Move image tensor to GPU if available
                            imag_vld = imag_vld.to(device)

                            # Move labels tensor to GPU if available
                            labl_vld = labl_vld.to(device)

                            # Get probability log from model
                            prob_log_vald = self.model(imag_vld)

                            # Calculate loss. Add validation loss
                            validation_loss += self.criterion(prob_log_vald, labl_vld)

                            # Calculate probability
                            prob = torch.exp(prob_log_vald)

                            # Get top probability and label
                            top_prob, top_label = prob.topk(1, dim=1)

                            # Compare model predection with actual label
                            model_label_comp = top_label == labl_vld.view(*top_label.shape)

                            #Calculate model accuracy
                            model_accuracy += torch.mean(model_label_comp.type(torch.FloatTensor))

                    # Set model to training
                    self.model.train()

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(step_loss/print_result_every),
                          "Validation Loss: {:.3f}.. ".format(validation_loss/len(valid_dataloader)),
                          "Validation Accuracy: {:.3f}".format(100 * (model_accuracy/len(valid_dataloader))))

                    step_loss = 0
        # Set model to evaliation
        self.model.eval()
        print("Done Training")

    def test_network(self, test_dataloader, device):
        ''' Test Network
        
        Arguments:
            test_dataloader: Test data set loader
            device: Device on which training should run. GPU or CUDA
        '''
        print("Running Test")
        validation_loss = 0
        model_accuracy = 0
        with torch.no_grad():
            self.model.to(device)
            self.model.eval()
            for imag_vld, labl_vld in test_dataloader:
                # Move image tensor to GPU if available
                imag_vld = imag_vld.to(device)

                # Move labels tensor to GPU if available
                labl_vld = labl_vld.to(device)

                # Caculate loss and accuracy
                prob_log_vald = self.model(imag_vld)
                validation_loss += self.criterion(prob_log_vald, labl_vld)
                prob = torch.exp(prob_log_vald)
                top_prob, top_label = prob.topk(1, dim=1)
                model_label_comp = top_label == labl_vld.view(*top_label.shape)
                model_accuracy += torch.mean(model_label_comp.type(torch.FloatTensor))

            print("Validation Loss: {:.3f}.. ".format(validation_loss/len(test_dataloader)),
                  "Validation Accuracy: {:.3f}".format(100*(model_accuracy/len(test_dataloader))),
                  "Validation Data: {:.3f}".format(len(test_dataloader)))

        print("Done Test")

    def predict(self, image_path, topk=5, device='cpu'):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
                
        Arguments:
            image_path: Path to image
            topk: Top probabilities to return 
            device: Device on which training should run. GPU or CUDA
        Returns:
            probs: Probabilities of top classes.
            classes: topk classes
        
        '''
        dataHelper = DataHelper()
        image = dataHelper.process_image(image_path)
        image = np.expand_dims(image, axis=0)
        image_tensor = torch.from_numpy(image).type(torch.FloatTensor).to(device)

        self.model.eval()
        self.model.to(device)
        with torch.no_grad():        
            output = self.model(image_tensor)
        ps = torch.exp(output)
        probs, classes_idx  = ps.topk(topk)
        idx_class_map = {v: k for k, v in self.model.class_idx_mapping.items()}
    
        classes_idx = classes_idx.data.cpu().numpy().squeeze()
        classes = [idx_class_map[index] for index in classes_idx]
        return probs.data.cpu().numpy().squeeze(), classes