"""
model repository init-cli
"""

from argparse import ArgumentParser
from argparse import Namespace

from openxlab.model.handler.download_file import download_metafile_template
from openxlab.types.command_type import BaseCommand


class Init(BaseCommand):
    """Initialize the generated metafile.yaml file."""

    def get_name(self) -> str:
        return "init"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.usage = (
            "openxlab model init [OPTIONS]\n"
            "Example:\n"
            "> openxlab model init --path='/local/to/path'"
        )
        parser.add_argument(
            "-p", "--path", required=False, help="The path where the generated files are stored."
        )
        parser.add_argument(
            "-a",
            "--all",
            default=False,
            required=False,
            help="The path where the generated files are stored.",
        )

    def take_action(self, parsed_args: Namespace) -> int:
        download_metafile_template(parsed_args.path, parsed_args.all)
        return 0
