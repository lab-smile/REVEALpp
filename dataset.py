import json
import random
import re

import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoTokenizer
from transformers import CLIPTokenizer, CLIPTextModel



def _transform(n_px):
    """
    Defines a transformation pipeline for image preprocessing.

    Args:
        n_px (int): The target image size for resizing.

    Returns:
        torchvision.transforms.Compose: A composed transformation pipeline.
    """
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),  # Resize image
        lambda image: image.convert("RGB"),  # Convert to RGB
        ToTensor(),  # Convert image to tensor
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # Normalize using COCO mean & std
    ])


def _normalize_caption(text: str) -> str:
    """Return a lightly normalised caption string.

    The retinal reports are often stored as comma separated phrases with
    inconsistent spacing (e.g. "cup to disc ratio,0.6 ,artery fractal").  The
    downstream BERT/GatorTron tokenizer already knows how to handle commas as
    standalone tokens, but it expects whitespace around punctuation to avoid
    fusing neighbouring words into a single subword piece.  We therefore strip
    the text, collapse consecutive whitespace, and ensure commas are followed
    by a single space.  This keeps the clinical phrasing intact while letting
    the tokenizer observe each phrase as an individual token sequence.
    """

    if not isinstance(text, str):
        text = str(text)

    # Replace newlines and repeated whitespace with a single space
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Ensure commas act as separators by enforcing a trailing space
    text = re.sub(r"\s*,\s*", ", ", text)

    return text


def _tokenizer(text):
    """
    Returns a tokenizer based on the specified text model.

    Args:
        text (str): The name of the text model (only 'gatortron' is supported).

    Returns:
        transformers.AutoTokenizer: A tokenizer instance.

    Notes:
        - Currently supports only 'gatortron' tokenizer.
    """
    if text == 'gatortron':
        return AutoTokenizer.from_pretrained('UFNLP/gatortronS')
        #return AutoTokenizer.from_pretrained("bert-base-uncased")


def normalize_df(df, columns):
    """
    Normalizes specified columns in a DataFrame using standard normalization.

    Args:
        df (pd.DataFrame): The DataFrame containing numerical columns.
        columns (list of str): List of column names to normalize.

    Returns:
        pd.DataFrame: A new DataFrame with normalized values.
    """
    df_normalized = df.copy()
    for column in columns:
        df_normalized[column] = (df[column] - df[column].mean()) / df[column].std()
    return df_normalized


def get_img_id_to_img_path(annotations, df):
    """
    Maps image IDs to their file paths.

    Args:
        annotations (dict): JSON annotation data containing image metadata.
        df (pd.DataFrame): DataFrame with image metadata.

    Returns:
        dict: Mapping from image ID to file path.
    """
    img_id_to_img_path = {}
    for img_info in annotations['images']:
        img_id = img_info['id']
        file_name = img_info['path']
        if img_info['file_name'] in df['Name'].tolist():
            img_id_to_img_path[img_id] = file_name
    return img_id_to_img_path


def get_img_id_to_img_name(annotations, df):
    """
    Maps image IDs to their file names.

    Args:
        annotations (dict): JSON annotation data containing image metadata.
        df (pd.DataFrame): DataFrame with image metadata.

    Returns:
        dict: Mapping from image ID to file name.
    """
    img_id_to_img_name = {}
    for img_info in annotations['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        if file_name in df['Name'].tolist():
            img_id_to_img_name[img_id] = file_name
    return img_id_to_img_name


def get_img_id_to_captions(annotations):
    """
    Maps image IDs to their corresponding captions.

    Args:
        annotations (dict): JSON annotation data containing captions.

    Returns:
        dict: Mapping from image ID to a list of captions.
    """
    img_id_to_captions = {}
    for caption_info in annotations['annotations']:
        img_id = caption_info['image_id']
        if img_id not in img_id_to_captions:
            img_id_to_captions[img_id] = []
        img_id_to_captions[img_id].append(caption_info['caption'])
    return img_id_to_captions


class Fundus_RiskFactor_Dataset(torch.utils.data.Dataset):
    """
    Custom dataset class for fundus images and risk factors.

    Loads images, associated textual descriptions, and image features.
    """
    def __init__(self, config, mode):
        """
        Initializes the dataset.

        Args:
            config (object): Configuration object containing paths and settings.
        """
        print('## Preparing the dataset. This may take some time. ##')

        # Load the dataset based on training or testing mode
        if mode == 'train':
            with open(config.train_json, "r") as f:
                self.json = json.load(f)
                
        elif mode == 'val':
            with open(config.val_json, "r") as f:
                self.json = json.load(f)        
        
        elif mode == 'test':
            with open(config.test_json, "r") as f:
                self.json = json.load(f)

        # Load image feature data from CSV
        image_df_path = config.image_feature_file
        self.image_df = pd.read_csv(image_df_path, index_col=0)

        # Extract images and annotations
        self.image = self.json['images']
        self.text = self.json['annotations']

        # Create mappings for image ID to file path, name, and captions
        self.img_id_to_filepath = get_img_id_to_img_path(self.json, self.image_df)
        self.img_id_to_filename = get_img_id_to_img_name(self.json, self.image_df)
        self.img_id_to_captions = get_img_id_to_captions(self.json)

        # List of image IDs in the dataset
        self.img_ids = list(self.img_id_to_filename.keys())

        # Define columns for text-based image features
        self.text_columns = [
            'CDR_vertical', 'CDR_horizontal', 'Fractal_dimension', 'Vessel_density',
            'Distance_tortuosity', 'Squared_curvature_tortuosity', 'Tortuosity_density',
            'Artery_Fractal_dimension', 'Artery_Vessel_density', 'Artery_Distance_tortuosity',
            'Artery_Squared_curvature_tortuosity', 'Artery_Tortuosity_density',
            'Vein_Fractal_dimension', 'Vein_Vessel_density', 'Vein_Distance_tortuosity',
            'Vein_Squared_curvature_tortuosity', 'Vein_Tortuosity_density'
        ]
        
        
        for col in self.text_columns:
            col_mean = self.image_df.loc[self.image_df[col] != -1, col].mean()
            self.image_df.loc[self.image_df[col] == -1, col] = col_mean

        # Normalize the image feature data
        self.image_df = normalize_df(self.image_df, self.text_columns)

        # Define transformations and tokenizer
        self.transform = _transform(224)  # Resize images to 224x224
        self.tokenizer = _tokenizer('gatortron')  # Load tokenizer

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The dataset size.
        """
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, text, image_features)
                - image (torch.Tensor): Preprocessed image.
                - text (str): Randomly selected caption.
                - image_features (torch.Tensor): Extracted numerical image features.
        """
        img_id = self.img_ids[idx]

        # Select a random caption for the image
        text = _normalize_caption(random.choice(self.img_id_to_captions[img_id]))

        # Retrieve file paths and image names
        img_path = self.img_id_to_filepath[img_id]
        img_name = self.img_id_to_filename[img_id]

        # Load and transform the image
        image = Image.open(img_path)
        image = self.transform(image)

        # Extract and convert image features to a tensor
        image_features = self.image_df[self.image_df['Name'] == img_name].drop('Name', axis=1)[self.text_columns].values
        image_features = torch.tensor(image_features.tolist()[0], dtype=torch.float32)

        return image, text, image_features
