"""
get the file list of the dataset repository
"""
from argparse import ArgumentParser
from argparse import Namespace

from openxlab.dataset.handler.list_dataset_repository import query
from openxlab.types.command_type import BaseCommand


class Ls(BaseCommand):
    """List dataset repository resources."""

    def get_name(self) -> str:
        return "ls"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.usage = (
            "openxlab dataset ls [OPTIONS]\n\n"
            "List dataset repository resources. \n"
            "Note: if you are not log in, you can only get the list of public dataset repository.\n\n"
            "Example:\n"
            "> openxlab dataset ls --dataset-repo \"username/dataset-repo-name\""
        )
        parser.add_argument(
            "-r",
            "--dataset-repo",
            type=str,
            required=True,
            help="The address of dataset repository. format: username/dataset-repo-name.[required]",
        )

    def take_action(self, parsed_args: Namespace) -> int:
        query(dataset_repo=parsed_args.dataset_repo)
        return 0
