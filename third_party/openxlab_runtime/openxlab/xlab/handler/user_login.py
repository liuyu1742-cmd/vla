import os

from openxlab.xlab.handler.user_config import UserConfig, get_config_path
from openxlab.xlab.handler.user_token import get_token_via_api
import logging

logger = logging.getLogger("openxlab")


def login(ak, sk, re_login=False, relogin=False):
    if os.path.exists(get_config_path()) and re_login is False and relogin is False:
        logger.warning("AK and SK have been configured. You can set relogin as true to force a relogin.")
    get_token_via_api(ak, sk)
    user_config = UserConfig(ak, sk)
    user_config.store_to_local()
