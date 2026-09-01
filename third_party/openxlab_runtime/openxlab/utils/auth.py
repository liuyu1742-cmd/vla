import json
import os

from openxlab.config import const


def get_token() -> str:
    raise Exception(
        "This method has been discarded, please use get_jwt method in openxlab.xlab.handler.user_token to "
        "get token, see ReadMe for details"
    )


def check_login():
    """check user login or not. If user logged in, return true, else return false.
    Used in ls or info dataset repository command"""
    try:
        config_path = os.path.join(const.DEFAULT_CONFIG_DIR, const.DEFAULT_CLI_CONFIG_FILE_NAME)
        if os.path.exists(config_path):
            with open(config_path, encoding='utf-8') as f:
                config_content = f.read()
            config_dict = json.loads(config_content)
            if config_dict['ak'] and config_dict['sk']:
                return True
    except Exception:
        pass
    return False
