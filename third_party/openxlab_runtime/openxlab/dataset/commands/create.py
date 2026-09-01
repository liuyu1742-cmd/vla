"""
create dataset repository-cli
"""
from argparse import ArgumentParser
from argparse import Namespace

from openxlab.dataset.handler.create_dataset_repository import create_repo
from openxlab.dataset.utils import str2bool
from openxlab.types.command_type import BaseCommand


class Create(BaseCommand):
    """Create a dataset repository."""

    def get_name(self) -> str:
        return "create"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.usage = (
            'openxlab dataset create [OPTIONS]\n\n'
            "Create a dataset repository.\n\n"
            'Example:\n'
            '> openxlab dataset create --repo-name \"dataset_repo_name\"'
        )
        parser.add_argument(
            "--repo-name",
            required=True,
            help='The name of dataset repository.[required]',
        )
        parser.add_argument(
            "-p",
            "--private",
            type=str2bool,
            # default=False,
            # required=True,
            help='The visibility permission of repository.',
        )

    def take_action(self, parsed_args: Namespace) -> int:
        create_repo(repo_name=parsed_args.repo_name, private=parsed_args.private)

        return 0
