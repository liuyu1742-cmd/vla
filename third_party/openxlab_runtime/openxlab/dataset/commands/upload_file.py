"""
upload file to dataset repository
"""

from argparse import ArgumentParser
from argparse import Namespace

from openxlab.dataset.handler.upload_dataset_file import upload_file
from openxlab.types.command_type import BaseCommand


class UploadFile(BaseCommand):
    """Upload file from local to remote."""

    def get_name(self) -> str:
        return "upload-file"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.usage = (
            "openxlab dataset upload-file [OPTIONS]\n\n"
            "Upload file from local to remote.\n\n"
            "Example:\n"
            '> openxlab dataset upload-file --dataset-repo \"username/dataset-repo-name\" '
            '--source-path \"/path/to/local/file\" --target-path \"/raw/file\"'
        )
        parser.add_argument(
            "-r",
            "--dataset-repo",
            required=True,
            help="The address of dataset repository. format: username/dataset-repo-name.[required]",
        )
        parser.add_argument(
            "-s",
            "--source-path",
            type=str,
            required=True,
            help=("The local path of the file to upload.[required]"),
        )
        parser.add_argument(
            "-t",
            "--target-path",
            type=str,
            # required=True,
            help=("The target path to upload file.[optional]"),
        )
        parser.add_argument(
            "--upload_network",
            type=str,
            help=(
                "upload_network [cdn|vpc]. Specifies the network type for file upload, "
                "cdn (default): Use Content Delivery Network for faster distribution. "
                "vpc: Use Virtual Private Cloud for enhanced security and privacy.[optional]"
            ),
        )

    def take_action(self, parsed_args: Namespace) -> int:
        upload_file(
            dataset_repo=parsed_args.dataset_repo,
            source_path=parsed_args.source_path,
            target_path=parsed_args.target_path,
            upload_network=parsed_args.upload_network,
        )

        return 0
