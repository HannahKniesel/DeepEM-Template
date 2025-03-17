from abc import ABC, abstractmethod
from typing import Any, List
import os
from pathlib import Path
import datetime
import torch

from deepEM.Utils import print_error, print_info, print_warning


class AbstractInference(ABC):
    """
    Abstract base class for performing model inference. 

    Subclasses must implement all abstract methods to define how models are loaded, 
    how predictions are made, and how results are saved.
    """

    def __init__(self, model_path: str, data_path: str, batch_size: int) -> None:
        """
        Initializes the inference pipeline with model and data paths.

        Args:
            model_path (str): Path to the model checkpoint file or directory containing it.
            data_path (str): Path to the input data (single file or directory).
            batch_size (int): Number of samples to process in a single batch during inference.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self.find_model_file(model_path)
        self.data_path = data_path
        self.batch_size = batch_size

        if self.model_path:
            self.metadata = self.load_metadata()
            self.model = self.setup_model()
            self.load_checkpoint()

            # Create a directory for storing inference results
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_root = Path(self.model_path).parent.parent.parent.stem
            results_dir = (
                os.path.join(self.data_path, f"results-{model_root}", timestamp)
                if os.path.isdir(self.data_path)
                else os.path.join(Path(self.data_path).parent, f"results-{model_root}", timestamp)
            )
            self.save_to = results_dir
            os.makedirs(self.save_to, exist_ok=True)

            # Log model and data paths
            with open(os.path.join(self.save_to, "model-and-data.txt"), "w") as file:
                file.write(f"Model path: {os.path.abspath(self.model_path)}\n")
                file.write(f"Data path: {os.path.abspath(self.data_path)}\n")

    def find_model_file(self, input_path: str) -> str:
        """
        Finds the model checkpoint file (`best_model.pth`) in the given path.

        If `input_path` is a file, checks if it's named "best_model.pth". 
        If `input_path` is a directory, searches recursively for "best_model.pth".

        Args:
            input_path (str): Path to a model file or directory containing it.

        Returns:
            str: Absolute path to the model checkpoint if found, else None.
        """
        if os.path.isfile(input_path):
            if os.path.basename(input_path) == "best_model.pth":
                print_info(f"Found model checkpoint at {input_path}")
                return input_path
            elif(input_path.lower().endswith(('.pth', '.pt'))):
                print_error("Provided file is no .pth or .pt file.")
                return None
            else:
                print_warning("Provided file is not named 'best_model.pth'. Expected 'best_model.pth'.")
                return input_path
        elif os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                if "best_model.pth" in files:
                    model_file = os.path.join(root, "best_model.pth")
                    if("TrainingRun" in model_file):
                        print_info(f"Found model checkpoint at {model_file}")
                        return model_file
            print_error("No 'best_model.pth' was found for a TrainingRun within the provided directory.")
            return None
        else:
            print_error("Invalid model path: not a file or directory.")
            return None

    def load_metadata(self) -> dict:
        """
        Loads metadata from the model checkpoint.

        Returns:
            dict: Metadata extracted from the checkpoint.
        """
        checkpoint = torch.load(self.model_path)
        return checkpoint["metadata"]

    def load_checkpoint(self) -> None:
        """
        Loads model weights and sets the model to evaluation mode.
        """
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model.to(self.device)
        self.metadata = checkpoint["metadata"]

    @abstractmethod
    def setup_model(self) -> torch.nn.Module:
        """
        Defines and initializes the model architecture for inference.

        Returns:
            torch.nn.Module: The model ready for inference.
        """
        raise NotImplementedError("The 'setup_model' method must be implemented by the DL specialist.")

    def get_image_files(self, folder_path: str, ext:List[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif")) -> List[str]:
        """
        Retrieves all image files from a directory.

        Args:
            folder_path (str): Path to the directory containing images.

        Returns:
            List[str]: List of file paths to images in the directory.
        """
        return [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.lower().endswith(ext)
        ]

    def inference(self) -> None:
        """
        Runs inference on the input data.

        Depending on whether `data_path` is a file or directory, 
        calls `predict_single()` for a single image or `predict_batch()` for multiple images.
        """
        with torch.no_grad():
            if self.model_path:
                if os.path.isdir(self.data_path):
                    self.predict_batch()
                elif os.path.isfile(self.data_path):
                    self.predict_single()
                else:
                    print_error(f"Invalid data path: {self.data_path} is neither a file nor a directory.")

    @abstractmethod
    def predict_single(self) -> Any:
        """
        Performs inference on a single image.

        Implementations should call `save_prediction(prediction, save_file)` 
        to store the prediction result.

        Returns:
            Any: The model's prediction for the given input.
        """
        raise NotImplementedError("The 'predict_single' method must be implemented by the DL specialist.")

    @abstractmethod
    def predict_batch(self) -> List[Any]:
        """
        Performs inference on a batch of images.

        Implementations should call `save_prediction(prediction, save_file)` 
        for each predicted output.

        Returns:
            List[Any]: List of predictions for the batch.
        """
        raise NotImplementedError("The 'predict_batch' method must be implemented by the DL specialist.")

    @abstractmethod
    def save_prediction(self, prediction: Any, save_file: str) -> None:
        """
        Saves a model prediction to a file.

        Args:
            prediction (Any): The prediction result to be saved.
            save_file (str): Path to the file where the prediction should be stored.
        """
        raise NotImplementedError("The 'save_prediction' method must be implemented by the DL specialist.")
