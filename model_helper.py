import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models
from flower_classifier import FlowerClassifier

class ModelHelper:    
    def save_checkpoint(self, classifier_model, epochs, fileName):
        ''' Save checkpoint

        Arguments:
            classifier_model: Trained FlowerClassifier model object
            epochs: Number of iterations used for training
            fileName: Name of file to save model

        ''' 
        checkpoint = {'arch': classifier_model.arch,
                      'hidden_units': classifier_model.hidden_units,
                      'drop_p': classifier_model.drop_p,                      
                      'lr' : classifier_model.lr,
                      'state_dict': classifier_model.model.state_dict(),
                      'classifier': classifier_model.model.classifier,             
                      'optimizer_dict': classifier_model.optimizer.state_dict(),
                      'class_idx_mapping': classifier_model.model.class_idx_mapping,
                      'epochs' : epochs}

        torch.save(checkpoint, fileName)
        
    def load_model(self, checkpoint_filename):
        ''' Load saved Model

        Arguments:
            checkpoint_filename: Name of file
        Returns:
            model_saved: FlowerClassifier model object

        ''' 
        state = torch.load(checkpoint_filename)
        model_saved = FlowerClassifier(arch = state['arch'], hidden_units = state['hidden_units'],
                                       drop_p = state['drop_p'], lr = state['lr'], 
                                       class_idx_mapping = state['class_idx_mapping'])
        
        # update model classifier
        model_saved.classifier = state['classifier']
        model_saved.model.classifier = model_saved.classifier        
        model_saved.model.load_state_dict(state['state_dict'])
        return model_saved