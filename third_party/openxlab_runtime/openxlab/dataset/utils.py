import argparse
import hashlib
import os
from pathlib import Path
import re
import time
from typing import List
from urllib.parse import urlparse

import requests
import urllib3.exceptions


def parse_url(url: str):
    o = urlparse(url)
    host = f"{o.scheme}://{o.hostname}"
    if o.port is not None:
        host += ":" + str(o.port)

    dataset_name = o.path.split("/")[-1]
    return host, dataset_name


def get_api_token_from_env():
    return os.environ.get("OPENDATALAB-API-TOKEN", "")


SYMBOLS = {
    "customary": ("B", "K", "M", "G", "T", "P", "E", "Z", "Y"),
    "customary_ext": ("byte", "kilo", "mega", "giga", "tera", "peta", "exa", "zetta", "iotta"),
    "iec": ("Bi", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"),
    "iec_ext": ("byte", "kibi", "mebi", "gibi", "tebi", "pebi", "exbi", "zebi", "yobi"),
}


def bytes2human(n, format="%(value).1f%(symbol)s", symbols="customary"):
    """
    Convert n bytes into a human-readable string based on format.
    symbols can be either "customary", "customary_ext", "iec" or "iec_ext",
    see: http://goo.gl/kTQMs
    """

    # Ensure n is an integer
    n = int(n)
    
    # Handle special case for zero bytes
    if n == 0:
        symbols = SYMBOLS[symbols]
        return format % dict(symbol=symbols[0], value=0)
        
    if n < 0:
        raise ValueError("n < 0")
        
    symbols = SYMBOLS[symbols]
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i + 1) * 10
    
    # Ensure proper handling of very small values
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    
    return format % dict(symbol=symbols[0], value=n)


def human2bytes(s):
    """
    Attempts to guess the string format based on default symbols
    set and return the corresponding bytes as an integer.
    When unable to recognize the format ValueError is raised.
    """
    init = s
    num = ""
    while s and s[0:1].isdigit() or s[0:1] == ".":
        num += s[0]
        s = s[1:]
    num = float(num)
    letter = s.strip()
    for name, sset in SYMBOLS.items():
        if letter in sset:
            break
    else:
        if letter == "k":
            # treat 'k' as an alias for 'K' as per: http://goo.gl/kTQMs
            sset = SYMBOLS["customary"]
            letter = letter.upper()
        else:
            raise ValueError("can't interpret %r" % init)
    prefix = {sset[0]: 1}
    for i, s in enumerate(sset[1:]):
        prefix[s] = 1 << (i + 1) * 10
    return int(num * prefix[letter])


def format_progress_string(progress, idx, total_files, finished_size, total_size):
    return (f"Total progress: {progress}%, total files:{idx+1}/{total_files}, "
            f"downloaded size: {bytes2human(finished_size)}, total size: {bytes2human(total_size)}")


def str2bool(v):
    """convert keywords like true to True, so do as False"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def find_url(text: str) -> List[str]:
    """Find all URLs in a given text using regular expressions.

    Args:
        text (str): The input text to search for URLs.

    Returns:
        List[str]: A list of URLs found in the text.

    """
    urls = re.findall(r'(https?://[^"]+)', text)
    return urls


def replace_urls(text: str, urls: list) -> str:
    """Replace URLs in a given text with a specified prefix and suffix using the rich package.

    Args:
        text (str): The input text to replace URLs in.
        urls (List[str]): A list of URLs to replace in the text.
        prefix (str, optional): The prefix to add before each URL. Defaults to "".
        suffix (str, optional): The suffix to add after each URL. Defaults to "".

    Returns:
        str: The modified text with URLs replaced.

    """
    for url in urls:
        new_url = "[blue underline]" + url + "[/blue underline]"
        text = text.replace(url, new_url)
    return text


def highlight_urls(text: str, prefix: str = "", suffix: str = ""):
    urls = find_url(text=text)
    if suffix:
        text = text.replace(urls[0], urls[0] + suffix)
        urls = [url + suffix for url in urls]
    new_text = replace_urls(text=text, urls=urls)
    return new_text


# buf_size = 128 kb chunksize
def calculate_file_sha256(file_path: str, buf_size: int = 131072):
    if not Path(file_path).is_file():
        raise Exception(f"file {file_path} does not exist")
    sha256_obj = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            sha256_obj.update(data)

    return sha256_obj.hexdigest()


def get_file_content(file_path: str, buf_size: int = 131072) -> str:
    """
    This function reads the content of a file at the specified file path and returns it as a bytes object.

    Args:
        file_path (str): The path to the file to be read.
        buf_size (int, optional): The buffer size used for reading the file. Defaults to 131072 (128 KB).

    Returns:
        bytes: The content of the file as a bytes object.

    Raises:
        Exception: If the specified file does not exist.

    Example:
        >>> file_content = get_file_content('/path/to/file.txt')
        >>> print(file_content)
        b'This is the content of the file.'
    """
    if not Path(file_path).is_file():
        raise Exception(f"file {file_path} does not exist")
    file_content = b""
    with open(file_path, "rb") as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            file_content += data
    file_content = file_content.decode("utf-8")
    return file_content


def retry_with_backoff(max_retries=3, base_delay=1, max_delay=32):
    """
    Decorator to retry a function in case of ConnectionError or TimeoutError, with exponential backoff.

    Args:
        max_retries (int, optional): Maximum number of retries. Defaults to 3.
        base_delay (int, optional): Initial delay in seconds. Defaults to 1.
        max_delay (int, optional): Maximum delay in seconds. Defaults to 32.

    Returns:
        function: Decorated function.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            retry_count = 0
            delay = base_delay
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.ConnectionError, 
                        requests.exceptions.ReadTimeout, 
                        urllib3.exceptions.ReadTimeoutError):
                    # print(f"ConnectionError or TimeoutError: {e}")
                    time.sleep(delay)
                    retry_count += 1
                    delay = min(delay * 2, max_delay)

            # print(f"Function {func.__name__} failed after {max_retries} retries.")
            raise Exception(
                f"requests.exceptions.ConnectionError or TimeoutError: function {func.__name__} failed after {max_retries} retries."
            )

        return wrapper

    return decorator
