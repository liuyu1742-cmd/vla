"""
download the repository of dataset
"""
from argparse import ArgumentParser
from argparse import Namespace

from openxlab.dataset.handler.get_dataset_repository import get
from openxlab.types.command_type import BaseCommand


class Get(BaseCommand):
    """Get the dataset repository."""

    def get_name(self) -> str:
        return "get"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.usage = (
            "openxlab dataset get [OPTIONS]\n\n"
            "Get the dataset repository.\n\n"
            "Example:\n"
            ">openxlab dataset get --dataset-repo \"username/dataset-repo-name\" "
            "--target-path \"/path/to/local/folder\""
        )
        parser.add_argument(
            "-r",
            "--dataset-repo",
            type=str,
            required=True,
            help="The address of dataset repository. format: username/dataset-repo-name.[required]",
        )
        parser.add_argument(
            "-t",
            "--target-path",
            type=str,
            help="The target local path to save the dataset repository. [optional]",
        )

    def take_action(self, parsed_args: Namespace) -> int:
        get(parsed_args.dataset_repo, parsed_args.target_path)

        return 0
