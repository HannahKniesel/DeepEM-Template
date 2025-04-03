import tifffile
from PIL import Image
from typing import Any, List
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
from pathlib import Path


from deepEM.Inferencer import AbstractInference
from src.Model import Model 


class SimpleDataset(Dataset):
    
    def __init__(self, data_path):
        self.image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

        
        if(self.is_image_file(data_path)):
            self.data_paths = [data_path]
        else:

            # Define the directory and image extensions
            directory = Path(data_path)

            # Get all image files with matching extensions
            self.data_paths = [file for file in directory.iterdir() if file.suffix.lower() in self.image_extensions]

            if(len(self.data_paths)==0):
               print(f"WARNNG::Cannot process provided path {data_path}.") 
        
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Resize((32,32)),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    def is_image_file(self, file_path):
        if(os.path.isfile(file_path)):
            return os.path.splitext(file_path)[-1].lower() in self.image_extensions
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.data_paths[idx]).convert("RGB")
        return self.transform(img)              
            
        
    
class Inference(AbstractInference):
    """
    Class for model inference. Implements all abstract methods
    to handle loading models, making predictions, and saving results.
    """
    def __init__(self, model_path: str, data_path: str, batch_size: int) -> None:
        super().__init__(model_path, data_path, batch_size)

    def min_max_norm(self, value):
        return (value-value.min())/(value.max()-value.min())
        
    def setup_model(self) -> None:
        """
        sets up the model class for inference.

        Returns: 
            torch.nn.Module: the model
        """
        return Model().to(self.device)
     
    
    def predict_single(self) -> Any:
        """
        Perform inference on a single image.

        Returns:
            Any: The prediction result for the image.
        """
        dataset = SimpleDataset(self.data_path)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size, 
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=4,
        )
        with torch.no_grad():
       
            for img in dataloader:
                img = img.to(self.device)
                outputs = self.model(img).detach().cpu()
                _, predicted = torch.max(outputs, 1)
                prediction = {"image": img.squeeze().cpu(), 
                    "prediction": self.metadata["class_names"][predicted]}
                self.save_prediction(prediction,  os.path.join(self.save_to, f"{Path(self.data_path).stem}_pred"))
                
            
        
        return 
        
    
    def predict_batch(self) -> List[Any]:
        """
        Perform inference on a batch of images.
        """
        dataset = SimpleDataset(self.data_path)
        dataloader = DataLoader(
            dataset,
            batch_size=16, 
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=4,
        )
        
        with torch.no_grad():
            index = 0
            for img in dataloader:
                img = img.to(self.device)
                outputs = self.model(img).detach().cpu()
                _, predicted = torch.max(outputs, 1)
                prediction = [{"image": i.squeeze().cpu(), "prediction": self.metadata["class_names"][p]} for i,p in zip(img,predicted)]
                [self.save_prediction(p,  os.path.join(self.save_to, f"prediction_{str(index+i).zfill(4)}")) for i,p in enumerate(prediction)]
                index += img.shape[0]
        return 
        

    

    def save_prediction(self, prediction, save_file: str) -> None:
        """
        Save predictions to a file.

        Args:
            input (Any): single input to save.
            prediction (Any): Prediction of the input to save. (Return of the self.predict_single method)
            save_file (str): Filename and Path to save the predictions. You need to set the format.
        """
        fig = plt.figure()
        plt.imshow(self.min_max_norm(prediction["image"].permute(1,2,0)))
        plt.title(f"Prediction: {prediction['prediction']}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{save_file}.jpg")
        plt.close()
        print(f"Saved prediction to {save_file}")


    
    