from flower_classifier import FlowerClassifier
from model_helper import ModelHelper
import argparse
import torch
import json
from prettytable import PrettyTable

#python predict.py flowers/test/52/image_04200.jpg trained_models/vgg16_trained_model.pth --top_k 5 --category_names cat_to_name.json
#python predict.py flowers/test/52/image_04200.jpg trained_models/densenet121_trained_model.pth --top_k 5 --category_names cat_to_name.json

def main():
    # Define input args
    parser = argparse.ArgumentParser(
    description='Flower Class Predictor Options',
    )
    
    parser.add_argument("image_path", help="Path to Image")
    
    parser.add_argument("checkpoint", help="Path to trained model checkpoint")
    
    parser.add_argument("--top_k", help="Top Predections. (default: 5)", 
                        default=5, type=int)
    
    parser.add_argument("--category_names", help="Catagory Names JSON.",
                        default="cat_to_name.json")
    
    
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
    
    # Load trained model
    model_helper = ModelHelper()    
    saved_model = model_helper.load_model(checkpoint_filename=args["checkpoint"])
    
    # Predict flower class
    prob, label = saved_model.predict(image_path=args["image_path"], topk=args["top_k"], device=device)
    
    # Load Label Names
    with open(args["category_names"], 'r') as f:
        cat_to_name = json.load(f)
    label_names = [cat_to_name[key] for key in label ]
    
    # Print Result
    result = PrettyTable()    
    result.field_names = ["Flower Class Name", "Probability"]
    
    for label,pb in zip(label_names, prob):
        result.add_row([label, pb])
    
    print(result)
    
if __name__ == '__main__':
    main()