"""
remove the dataset repository
"""
from argparse import ArgumentParser
from argparse import Namespace

from openxlab.dataset.handler.remove_dataset_repository import remove_repo
from openxlab.types.command_type import BaseCommand


class Remove(BaseCommand):
    """Remove a dataset repository."""

    def get_name(self) -> str:
        return "remove"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.usage = (
            "openxlab dataset remove [OPTIONS]\n\n"
            "Remove a dataset repository.\n\n"
            "Example:\n"
            "> openxlab dataset remove --dataset-repo \"username/dataset-repo-name\""
        )
        parser.add_argument(
            "-r",
            "--dataset-repo",
            help="The address of dataset repository. format: username/dataset-repo-name.[required]",
        )

    def take_action(self, parsed_args: Namespace) -> int:
        remove_repo(dataset_repo=parsed_args.dataset_repo)

        return 0
