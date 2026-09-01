from argparse import ArgumentParser
from argparse import Namespace

from openxlab.config import version as config_version
from openxlab.dataset.commands.commit import Commit
from openxlab.dataset.commands.create import Create
from openxlab.dataset.commands.download import Download

# from openxlab.dataset.commands.download import Download
from openxlab.dataset.commands.get import Get
from openxlab.dataset.commands.info import Info
from openxlab.dataset.commands.ls import Ls
from openxlab.dataset.commands.remove import Remove
from openxlab.dataset.commands.upload_file import UploadFile
from openxlab.dataset.commands.upload_folder import UploadFolder
from openxlab.dataset.commands.visibility import Visibility
from openxlab.dataset.handler.commit_dataset_info import commit
from openxlab.dataset.handler.create_dataset_repository import create_repo
from openxlab.dataset.handler.download_dataset_repository import download
from openxlab.dataset.handler.get_dataset_repository import get
from openxlab.dataset.handler.info_dataset_repository import info
from openxlab.dataset.handler.list_dataset_repository import query
from openxlab.dataset.handler.remove_dataset_repository import remove_repo
from openxlab.dataset.handler.upload_dataset_file import upload_file
from openxlab.dataset.handler.upload_dataset_folder import upload_folder
from openxlab.dataset.handler.visible_dataset_repository import visibility
from openxlab.types.command_type import BaseCommand


def help():
    print("help")


class Dataset(BaseCommand):
    """Dataset Demo"""

    sub_command_list = [
        Get,
        Create,
        UploadFile,
        UploadFolder,
        Info,
        Ls,
        Commit,
        Download,
        Remove,
        Visibility,
    ]

    def get_name(self) -> str:
        return "dataset"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.usage = "openxlab dataset [OPTIONS]\n\n"
        # parser.add_argument(
        #     "--test",
        #     help=(" This argument is a test argument"),
        # )
