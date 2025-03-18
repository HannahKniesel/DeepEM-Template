import ipywidgets as widgets
import json
import os

def find_file(root_dir, filename):
    for dirpath, _, filenames in os.walk(root_dir):
        if filename in filenames:
            return os.path.join(dirpath, filename)  # Return the first match
    return None  # If the file is not found

def find_model_file(input_path: str) -> str:
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
    
def format_time(seconds):
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    time_str = f"{int(days)}d{int(hours)}h{int(minutes)}m{int(seconds)}s" if days > 0 else f"{int(hours)}h{int(minutes)}m{int(seconds)}s"
    return time_str

def print_info(text):
    print("[INFO]::"+text)
    
def print_error(text):
    print("[INFO]::"+text)
    
def print_warning(text):
    print("[INFO]::"+text)

   
def create_text_widget(name,value,description):
    text_widget = widgets.Text(
        value=str(value),
        description=name,
        style={'description_width': 'initial'}
    )
    description_widget = widgets.HTML(value=f"<b>Hint:</b> {description}")
        
    return (text_widget, description_widget)


def load_json(file):
    # Open and load the JSON file
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def extract_defaults(config):
    defaults = {}
    for key, value in config.items():
        if isinstance(value, dict) and "default" in value:
            # Extract default value from nested hyperparameter dict
            defaults[key] = value["default"]
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            nested_defaults = extract_defaults(value)
            defaults.update(nested_defaults)
        else:
            pass
    return defaults

def get_fixed_parameters(config_file):
    params_json = load_json(config_file)["parameter"]
    fixed_parameter = {}
    for k in params_json.keys(): 
        fixed_parameter[k] = params_json[k]["value"]
        
    return fixed_parameter
        




