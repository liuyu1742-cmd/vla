"""
download dataset file|folder cli
"""
from argparse import ArgumentParser
from argparse import Namespace

from openxlab.dataset.handler.download_dataset_repository import download
from openxlab.types.command_type import BaseCommand


class Download(BaseCommand):
    """Download file or folder of a dataset repository."""

    def get_name(self) -> str:
        return "download"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.usage = (
            "openxlab dataset download [options]\n\n"
            "Download file or folder of a dataset repository.\n\n"
            "Example:\n"
            "> openxlab dataset download --dataset-repo \"username/dataset-repo-name\" "
            "--source-path \"/raw/file\" --target-path \"/path/to/local/folder\""
        )
        parser.add_argument(
            "-r",
            "--dataset-repo",
            type=str,
            required=True,
            help="The address of dataset repository. format: username/dataset-repo-name.[required]",
        )
        parser.add_argument(
            "-s",
            "--source-path",
            type=str,
            required=True,
            help="The relative path of the target file or folder to download.[required]",
        )
        parser.add_argument(
            "-t",
            "--target-path",
            type=str,
            required=False,
            help="The target local path to store the file or folder.[optional]",
        )

    def take_action(self, parsed_args: Namespace) -> int:
        download(
            dataset_repo=parsed_args.dataset_repo,
            source_path=parsed_args.source_path,
            target_path=parsed_args.target_path,
        )

        return 0
