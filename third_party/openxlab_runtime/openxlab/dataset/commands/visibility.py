"""
setting dataset repository visibility-cli
"""

from argparse import ArgumentParser
from argparse import Namespace

from openxlab.dataset.handler.visible_dataset_repository import visibility
from openxlab.dataset.utils import str2bool
from openxlab.types.command_type import BaseCommand


class Visibility(BaseCommand):
    """Set dataset repository visibility to public or private"""

    def get_name(self):
        return "visibility"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.usage = (
            'openxlab dataset visibility [OPTIONS]\n\n'
            "set dataset visibility.\n\n"
            'Example:\n'
            '> openxlab dataset visibility --dataset-repo \"username/dataset-repo-name\" --private True'
        )
        parser.add_argument(
            "-r",
            "--dataset-repo",
            required=True,
            help='The address of dataset repository. format: username/dataset-repo-name.[required]',
        )
        parser.add_argument(
            "-p",
            "--private",
            type=str2bool,
            # default=True,
            required=True,
            help='The visibility permission of repository.[required]',
        )

    def take_action(self, parsed_args: Namespace) -> int:
        visibility(parsed_args.dataset_repo, parsed_args.private)

        return 0
