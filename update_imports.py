#!/usr/bin/env python
'''
Utility script to update import statements throughout the codebase
to use the new ANTsPy adapter instead of direct NiftyPET imports.
'''

__author__ = "devhliu"
__copyright__ = "Copyright 2025-03-24"

import os
import re
from pathlib import Path

def update_file(file_path):
    """Update import statements in a file
    
    Arguments:
    - file_path: Path to the file to update
    
    Returns:
    - True if file was modified, False otherwise
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace direct imports of nimpa
    original_content = content
    content = re.sub(
        r'from niftypet import nimpa',
        'from amypet.antspy_adapter import nimpa',
        content
    )
    
    # Replace direct imports of spm12
    content = re.sub(
        r'import spm12',
        'from amypet.antspy_adapter import spm12',
        content
    )
    
    # Check if content was modified
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

def main():
    """Main function to update import statements in all Python files"""
    # Get the root directory of the project
    root_dir = Path(__file__).parent
    
    # Files to skip
    skip_files = [
        'ants_compat.py',
        'ants_utils.py',
        'antspy_adapter.py',
        'update_imports.py'
    ]
    
    # Count of modified files
    modified_count = 0
    
    # Walk through all Python files in the project
    for root, dirs, files in os.walk(root_dir):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
        
        for file in files:
            if file.endswith('.py') and file not in skip_files:
                file_path = os.path.join(root, file)
                if update_file(file_path):
                    modified_count += 1
                    print(f"Updated: {file_path}")
    
    print(f"\nTotal files modified: {modified_count}")

if __name__ == '__main__':
    main()