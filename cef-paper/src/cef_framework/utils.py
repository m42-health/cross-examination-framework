"""
Module containing utility functions used through project

Author: Hamza Javed
Company: M42
Date: April 2024
"""

# Import relevant packages
import yaml
from pathlib import Path
from jsonfinder import jsonfinder
import json


#--
# File reading

def load_yaml(yaml_filepath: str | Path) -> dict:
    """
    Read in YAML files

    Args:
        yaml_filepath: filepath to yaml file to be read

    Returns:
        Contents of the YAML file
    """
    try:
        with open(yaml_filepath, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {yaml_filepath}")
    except IOError as e:
        raise IOError(f"Error reading file: {e}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")



def load_text(text_filepath: str | Path) -> str:
    """
    Read in text file

    Args:
        text_filepath: filepath to text file to be read

    Returns:
        Contents of the text file
    """
    try:
        with open(text_filepath, 'r') as file:
            data = file.read()
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {text_filepath}")
    except Exception as e:
        raise Exception(f"Error: {e}")

def safe_load_questions(s):
    hits = list(jsonfinder(s))
    for hit in hits:
        json_content = hit[2]
        if isinstance(json_content, list) and len(json_content) > 0:
            flag = 0
            for x in json_content:
                if not isinstance(x, dict) or "question" not in x or "answer" not in x:
                    flag = 1
                    break
            if flag:
                continue
            else:
                return json.dumps(json_content)
    return "None"