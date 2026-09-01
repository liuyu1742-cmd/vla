"""
upload folder to dataset repository
"""

from argparse import ArgumentParser
from argparse import Namespace

from openxlab.dataset.handler.upload_dataset_folder import upload_folder
from openxlab.types.command_type import BaseCommand


class UploadFolder(BaseCommand):
    """Upload folder from local to remote."""

    def get_name(self) -> str:
        return "upload-folder"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.usage = (
            "openxlab dataset upload-folder [OPTIONS]\n\n"
            "Upload folder from local to remote.\n\n"
            "Example:\n"
            '> openxlab dataset upload-folder  --dataset-repo "username/dataset-repo-name" '
            '--source-path "/path/to/local/folder" --target-path "/raw/folder"'
        )
        parser.add_argument(
            "-r",
            "--dataset-repo",
            type=str,
            required=True,
            help="The address of dataset repository. format: username/dataset-repo-name.[required]",
        )
        parser.add_argument(
            "-s",
            "--source-path",
            type=str,
            required=True,
            help=("The local path of the folder to upload.[required]"),
        )
        parser.add_argument(
            "-t",
            "--target-path",
            type=str,
            # required=True,
            help=("The target path to upload folder.[optional]"),
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
        upload_folder(
            dataset_repo=parsed_args.dataset_repo,
            source_path=parsed_args.source_path,
            target_path=parsed_args.target_path,
            upload_network=parsed_args.upload_network,
        )

        return 0
