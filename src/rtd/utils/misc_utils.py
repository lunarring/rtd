import os
import inspect

def get_repo_root(current_file=None):
    """
    Get the absolute path to the repository root directory.
    
    Args:
        current_file (str, optional): Path to the current file. If None, uses the caller's file.
            This is useful when you want to find the repo root relative to a specific file.
    
    Returns:
        str: Absolute path to the repository root directory.
    """
    if current_file is None:
        # Use the caller's file location
        current_file = inspect.getframeinfo(inspect.currentframe().f_back).filename
    
    # Start from the current file's directory
    current_dir = os.path.dirname(os.path.abspath(current_file))
    
    # Go up the directory tree until we find a .git directory or reach the filesystem root
    while current_dir and not os.path.isdir(os.path.join(current_dir, '.git')):
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached the filesystem root
            # If we can't find a .git directory, return the original directory
            return os.path.dirname(os.path.abspath(current_file))
        current_dir = parent_dir
    
    return current_dir

def get_repo_path(relative_path, current_file=None):
    """
    Get the absolute path for a path that is relative to the repository root.
    
    Args:
        relative_path (str): Path relative to the repository root.
        current_file (str, optional): Path to the current file. If None, uses the caller's file.
            This is useful when you want to find paths relative to a specific file's repo.
        
    Returns:
        str: Absolute path.
    """
    return os.path.join(get_repo_root(current_file), relative_path) 