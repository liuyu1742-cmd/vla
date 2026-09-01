from argparse import Namespace
import getpass

from openxlab.types.command_type import BaseCommand
from openxlab.xlab.handler.user_config import UserConfig


class Config(BaseCommand):
    """Config Openxlab AK SK"""

    def get_name(self) -> str:
        return "config"

    def take_action(self, parsed_args: Namespace) -> int:
        ak = input("OpenXLab Access Key ID [NONE]: ")
        if len(ak) == 0:
            raise ValueError("access key id must not be empty")
        sk = getpass.getpass("OpenXLab Secret Access Key [None]: ")
        if len(sk) == 0:
            raise ValueError("secret access key must not be empty")
        user_config = UserConfig(ak.strip(), sk.strip())
        user_config.store_to_local()
        return 0
