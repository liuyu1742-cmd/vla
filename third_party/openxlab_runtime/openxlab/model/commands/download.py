"""
download model file|meta file|log file|readme file-cli
"""

from argparse import ArgumentParser
from argparse import Namespace

from openxlab.model import download
from openxlab.types.command_type import BaseCommand


class Download(BaseCommand):
    """Download model weight files."""

    def get_name(self) -> str:
        return "download"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.usage = (
            "openxlab model download [-h] -r MODEL_REPO -f FILE [-p PATH] [-o] {help} ...\n"
            "Example:\n"
            "> openxlab model download --model-repo=username/model_repo_name"
            "  --model_name=faster_rcnn \n"
            "  --output=/path/to/local/folder/\n"
            "  --overwrite=True"
        )
        parser.add_argument(
            "-r",
            "--model-repo",
            required=True,
            help="model repository address. format:username/repository.",
        )
        parser.add_argument(
            "-m", "--model-name", required=False, nargs="+", help="target model to be download."
        )
        parser.add_argument("-op", "--output", required=False, help="setting download path.")
        parser.add_argument(
            "-o", "--overwrite", default=False, type=bool, help="force overwriting local files."
        )
        parser.add_argument(
            "-i",
            "--ignore",
            required=False,
            help="ignore is used to filter files that do not need to be downloaded.",
        )

    def take_action(self, parsed_args: Namespace) -> int:
        download(
            parsed_args.model_repo,
            model_name=parsed_args.model_name,
            output=parsed_args.output,
            overwrite=parsed_args.overwrite,
            ignore=parsed_args.ignore,
        )
        return 0
