from flower_classifier import FlowerClassifier
from model_helper import ModelHelper
from data_util import DataHelper
import argparse
import torch
import os
from prettytable import PrettyTable

#python train.py flowers --arch vgg16 --save_dir trained_models --hidden_units 4096 --learning_rate 0.0001 --epochs 3 --dropout 0.5 --gpu
#python train.py flowers --arch densenet121 --save_dir trained_models --hidden_units 256 --learning_rate 0.001 --epochs 3 --dropout 0.25 --gpu

def main():
    # Define input args
    parser = argparse.ArgumentParser(
    description='Flower Classifier Options',
    )
    
    parser.add_argument("data_dir", help="Directory containing the dataset.",
                    default="flowers", nargs="?")
    
    INPUT_CHOSES_ARCH = ("vgg16", "vgg13", "densenet121")
    parser.add_argument("--arch", help="Model architecture. (default: vgg16)", choices=INPUT_CHOSES_ARCH,
                    default=INPUT_CHOSES_ARCH[0])
    
    parser.add_argument("--save_dir", help="Directory which will save trained models.",
                    default="trained_models")
    
    parser.add_argument("--hidden_units", help="Number of units in hidden layer. (default: 4096)",
                    default=4096, type=int)

    parser.add_argument("--learning_rate", help="Learning rate for Adam optimizer. (default: 0.001)",
                    default=0.0001, type=float)

    parser.add_argument("--epochs", help="Number of iterations over the dataset. (default: 3)",
                    default=3, type=int)    
    
    parser.add_argument("--dropout", help="Dropout Percentage 0 to 1. (default: 0.5)",
                    default=0.5, type=float)

    parser.add_argument("--gpu", help="Use GPU or CPU for training", default=False, action="store_true")
    
    # Parse input arguments
    args = vars(parser.parse_args())
    
    # Check GPU available if selected by user 
    available_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (args["gpu"] and not(torch.cuda.is_available())) :
        print("GPU not available can not continue.")
        return
    
    # Set device for training
    device = None
    if args["gpu"]:
        device = "cuda"
    else:
        device = "cpu"
    
    # Create checkpoint directory
    if(not(os.path.exists(args["save_dir"]))):
        os.system("mkdir -p " + args["save_dir"])
    
    # Print training parameters
    print("Training model with parameters::")
    input_param_table = PrettyTable()    
    input_param_table.field_names = ["Input Parameter", "Valie"]
    
    input_param_table.add_row(["Data Directory",args["data_dir"]])
    input_param_table.add_row(["Architecture",args["arch"]])
    input_param_table.add_row(["Trained Model Directory",args["save_dir"]])
    input_param_table.add_row(["Hidden Units",args["hidden_units"]])
    input_param_table.add_row(["Learning Rate",args["learning_rate"]])
    input_param_table.add_row(["Epochs",args["epochs"]])
    input_param_table.add_row(["Dropout",args["dropout"]])
    input_param_table.add_row(["GPU",args["gpu"]])
    print(input_param_table)
    print("")
    
    print("")
    
    # Create data loader
    dataHelper = DataHelper()
    train_datasets, test_datasets, valid_datasets, train_dataloader, test_dataloader, valid_dataloader = dataHelper.get_data_loder(data_dir=args["data_dir"])
    
    # Create FlowerClassifier object
    classifier = FlowerClassifier(arch=args["arch"], hidden_units=args["hidden_units"], drop_p=args["dropout"], lr=args["learning_rate"], class_idx_mapping=train_datasets.class_to_idx)    
    
    # Tran model on training data set    
    classifier.train(train_dataloader=train_dataloader, epochs=args["epochs"], valid_dataloader=valid_dataloader, print_result_every=20, device=device)
    
    print("")
    print("")
    print("-----------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------")
    
    # Test Trained Model
    classifier.test_network(test_dataloader=test_dataloader, device=device)    
    
    print("-----------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------")
    
    # Save model
    model_helper = ModelHelper()
    model_helper.save_checkpoint(classifier_model=classifier, epochs=args["epochs"], fileName=args["save_dir"]+"/"+args["arch"]+"_trained_model.pth")
    
    # Load model again to check if saving successfull
    saved_model = model_helper.load_model(checkpoint_filename=args["save_dir"]+"/"+args["arch"]+"_trained_model.pth")
    
if __name__ == '__main__':
    main()