"""
create model repository-cli
"""

from argparse import ArgumentParser
from argparse import Namespace

from openxlab.model import create
from openxlab.types.command_type import BaseCommand


class Create(BaseCommand):
    """Creating a model repository."""

    def get_name(self) -> str:
        return "create"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.usage = (
            "openxlab model create [OPTIONS]\n"
            "Example:\n"
            "> openxlab model create --model-repo='username/model_repo_name'"
        )
        parser.add_argument(
            "-r", "--model-repo", required=True, help="The name of model repository.[required]"
        )
        parser.add_argument(
            "-prt",
            "--private",
            type=bool,
            default=False,
            help="BOOLEAN  The visibility of the model repository. Default is False.",
        )
        parser.add_argument(
            "-s", "--source", type=str, required=True, help="The path of the meta file"
        )

    def take_action(self, parsed_args: Namespace) -> int:
        create(parsed_args.model_repo, parsed_args.source, parsed_args.private)
        return 0
