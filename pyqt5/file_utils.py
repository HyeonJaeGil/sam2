# file_utils.py

import os
import json
from PyQt5.QtWidgets import QFileDialog


class DirectoryCache:
    def __init__(self):
        self.cache_file = os.path.expanduser("~/.sam2_gui_cache.json")
        self.directories = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.directories, f)

    def get_cached_dir(self, key):
        return self.directories.get(key, "")

    def set_cached_dir(self, key, path):
        self.directories[key] = path
        self._save_cache()

    def ask_directory(self, parent, key, title="Select Directory"):
        default = self.get_cached_dir(key)
        dir_path = QFileDialog.getExistingDirectory(parent, title, default)
        if dir_path:
            self.set_cached_dir(key, dir_path)
        return dir_path
