"""Utility functions for data handling and caching."""

import pickle
import os


def read_stub(read_from_stub, stub_path):
    """
    Read cached data from a pickle file.
    
    Args:
        read_from_stub (bool): Whether to attempt reading from cache.
        stub_path (str): Path to the cache file.
    
    Returns:
        object: The cached data if available and valid, None otherwise.
    """
    if not read_from_stub or stub_path is None:
        return None
    
    if not os.path.exists(stub_path):
        return None
    
    try:
        with open(stub_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Warning: Failed to read stub from {stub_path}: {e}")
        return None


def save_stub(stub_path, data):
    """
    Save data to a pickle file for caching.
    
    Args:
        stub_path (str): Path where the cache file should be saved.
        data (object): The data to cache.
    """
    if stub_path is None:
        return
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        
        with open(stub_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Cached data saved to {stub_path}")
    except Exception as e:
        print(f"Warning: Failed to save stub to {stub_path}: {e}")
