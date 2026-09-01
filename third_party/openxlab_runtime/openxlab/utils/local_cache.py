import os.path
import time
import json

from openxlab.config import const


class LocalCache:
    def __init__(self, file_path):
        self.file_path = file_path
        self.cache = self.load_cache()

    def load_cache(self):
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_cache(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.cache, f)

    def set(self, key, value, expire=0):
        if expire > 0:
            expire += int(time.time())
        self.cache[key] = {'expire': expire, 'data': value}
        self.save_cache()

    def get(self, key):
        if key not in self.cache:
            return None
        if 0 < self.cache[key]['expire'] < int(time.time()):
            del self.cache[key]
            self.save_cache()
            return None
        return self.cache[key]['data']

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]
            self.save_cache()


cache = LocalCache(os.path.join(const.DEFAULT_CONFIG_DIR, 'cache.json'))
