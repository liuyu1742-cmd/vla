"""
get the information of dataset repository
"""
from argparse import ArgumentParser
from argparse import Namespace

from openxlab.dataset.handler.info_dataset_repository import info
from openxlab.types.command_type import BaseCommand


class Info(BaseCommand):
    """Get the information of a dataset repository."""

    def get_name(self) -> str:
        return "info"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.usage = (
            "openxlab dataset info [OPTIONS]\n\n"
            "Get the information of a dataset repository.\n"
            "Note: if you are not log in, you can only get the information of public dataset repository.\n\n"
            "Example:\n"
            "> openxlab dataset info --dataset-repo \"username/dataset-repo-name\""
        )
        parser.add_argument(
            "-r",
            "--dataset-repo",
            type=str,
            required=True,
            help="The address of dataset repository. format: username/dataset-repo-name.[required]",
        )

    def take_action(self, parsed_args: Namespace) -> int:
        info(dataset_repo=parsed_args.dataset_repo)
