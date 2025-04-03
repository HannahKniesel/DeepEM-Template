
import os
import torch
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image
import io

import torchvision
import torchvision.transforms as transforms

from deepEM.ModelTrainer import AbstractModelTrainer
from src.Model import Model 

criterion = torch.nn.CrossEntropyLoss()


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def min_max_norm(value):
        return (value - value.min())/(value.max()-value.min())

class ModelTrainer(AbstractModelTrainer):
    def __init__(self, data_path, logger, resume_from_checkpoint = None):
        """
        Initializes the trainer class for training, validating, and testing models.

        Args:
            model (torch.nn.Module): The model to train.
            logger (Logger): Logger instance for logging events.
            config (dict): Contains all nessecary hyperparameters for training. Must at least contain: `epochs`, `early_stopping_patience`, `validation_interval`, `scheduler_step_by`.
            resume_from_checkpoint (str): Path to resume checkpoint
            train_subset (float, optional): Use subset of training data. This can be used for quick hyperparamtertuning. Defaults to `None`. 
            reduce_epochs (float, optional): Use subset of epochs. This can be used for quick hyperparamtertuning. Defaults to `None`. 
        """
        super().__init__(data_path, logger, resume_from_checkpoint )
        
    def setup_model(self):
        """
        Setup and return the model for training, validation, and testing.

        This method must be implemented by the DL expert.

        Returns:
            model (lib.Model.AbstractModel): The dataloader for the training dataset.
        """
        return Model().to(self.device)

    
    def inference_metadata(self):
        """
        Returns possible metadata needed for inference (such as class names) as dictonary.
        This metadata will be saved along with model weights to the training checkpoints. 
        
        
        Returns:
            dict: dictonary with metadata
            
        """
        metadata = {}
        metadata["class_names"] = classes
        return metadata
        
            
    def setup_datasets(self):
        """
        Setup and return the dataloaders for training, validation, and testing.

        This method must be implemented by the DL expert.
        
        The data_path provided by the EM specialist can b accessed via self.data_path

        Returns:
            train_dataset (torch.utils.data.Dataset): The dataset for the training dataset.
            val_dataset (torch.utils.data.Dataset): The dataset for the validation dataset.
            test_dataset (torch.utils.data.Dataset): The dataset for the test dataset.
        """
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        train_dataset = torchvision.datasets.CIFAR10(root=self.data_path, train=True,
                                                download=True, transform=transform)
        
        val_dataset = torchvision.datasets.CIFAR10(root=self.data_path, train=False,
                                            download=True, transform=transform)
        test_dataset = val_dataset

        return train_dataset, val_dataset, test_dataset
    
    def setup_visualization_dataloaders(self, val_dataset, test_dataset):
        """
        Setup and return the dataloaders for visualization during validation, and testing.
        This method will subsample the val_dataset and test_dataset to contain self.parameter["images_to_visualize"] datapoints
        This method should be overidden for imbalanced data, to pick the most interesting data samples.
                        
        Inputs:
            valset (torch.utils.data.Dataset): The validation dataset.
            testset (torch.utils.data.Dataset): The test dataset.

        Returns:
            val_vis_loader (torch.utils.data.DataLoader): The dataloader for visualizing a subset of the validation dataset.
            test_vis_loader (torch.utils.data.DataLoader): The dataloader for visualizing a subset of the test dataset.
        """
        
        vis_val_subset = torch.utils.data.Subset(val_dataset, torch.randint(0,len(val_dataset), (self.parameter["images_to_visualize"],)))
        val_vis_loader = DataLoader(vis_val_subset, batch_size=self.parameter["batch_size"], shuffle=False)
        
        
        vis_test_subset = torch.utils.data.Subset(test_dataset, torch.randint(0,len(test_dataset), (self.parameter["images_to_visualize"],)))
        test_vis_loader = DataLoader(vis_test_subset, batch_size=self.parameter["batch_size"], shuffle=False)
        return val_vis_loader, test_vis_loader
          
        

    def setup_optimizer(self):
        """
        Setup and return the optimizer and learning rate scheduler.

        This method must be implemented by the DL expert.

        Returns:
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.parameter["learning_rate"], momentum=0.9)
        lr_scheduler = None
        return optimizer, lr_scheduler

    
    def compute_loss(self, outputs, targets):
        """
        Compute the loss for a batch.
        
        Args:
            outputs (torch.Tensor): Model outputs.
            targets (torch.Tensor): Ground truth labels.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        return criterion(outputs, targets)
        

    def train_step(self, batch_idx, batch):
        """
        Perform one training step.

        Args:
            batch (tuple): A batch of data (inputs, targets).
        
        Returns:
            torch.Tensor: The loss for this batch.
        """
        
        inputs, labels = batch

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.model(inputs.to(self.device))
        loss = self.compute_loss(outputs, labels.to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    
    
    def visualize(self, batch):
        """
        Visualizes the models input and output of a single batch and returns them as PIL.Image.

        Args:
            batch: A batch of data defined by your Dataset implementation.
        
        Returns:
            List[PIL.Image]: List of visualizations for the batch data.
            
        """
        self.model.eval()
        inputs, labels = batch
        outputs = self.model(inputs.to(self.device)).detach().cpu()
        _, predicted = torch.max(outputs, 1)

        
        fig,axs = plt.subplots(1, (inputs.shape[0]))
        
        pil_images = []
        for i, (input, output, gt) in enumerate(zip(inputs, predicted, labels)):
            # TODO plot annotated locations?            
            axs[i].imshow(min_max_norm(input).squeeze().permute(1,2,0))
            axs[i].set_title(f'Label: {classes[gt]:5s}\nPred: {classes[output]:5s}')
            axs[i].set_axis_off()
            
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG')
        buf.seek(0)  # Move the cursor to the beginning of the buffer
        pil_image = Image.open(buf)
        pil_images.append(pil_image)
        plt.close()
        return pil_images
    
    def compute_metrics(self, outputs, targets):
        """
        Computes a metric for evaluation based on the models outputs and targets

        Args:
            outputs[torch.Tensor]: A batch of model outputs
            targets[torch.Tensor]: A batch of targets
            
        
        Returns:
            dict: dictonary of the computed metrics
            
        """
        outputs = outputs.cpu()
        targets = targets.cpu()
        _, predicted = torch.max(outputs, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        accuracy = 100 * correct // total
        
        metrics = {"Acc": accuracy}
        
        return metrics
        

    def val_step(self, batch_idx, batch):
        """
        Perform one validation step.

        Args:
            batch (tuple): A batch of data (inputs, targets).
        
        Returns:
            torch.Tensor: The loss for this batch.
            dict: Dictionary of metrics for this batch (e.g., accuracy, F1 score, etc.).
            
        """
        self.model.eval()
        images, labels = batch
        outputs = self.model(images.to(self.device))
        loss = self.compute_loss(outputs, labels.to(self.device))
        metrics = self.compute_metrics(outputs, labels)
        return loss.item(), metrics
        

    def test_step(self, batch_idx, batch):
        """
        Perform one test step.

        Args:
            batch (tuple): A batch of data (inputs, targets).
        
        Returns:
            torch.Tensor: The loss for this batch.
            dict: Dictionary of metrics for this batch (e.g., accuracy, F1 score, etc.).
            
        """
        # Implementation could look like this:
        return self.val_step(batch_idx, batch)

    