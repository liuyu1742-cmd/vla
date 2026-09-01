"""
commit message to dataset repository cli
"""
from argparse import ArgumentParser
from argparse import Namespace

from openxlab.dataset.handler.commit_dataset_info import commit
from openxlab.types.command_type import BaseCommand


class Commit(BaseCommand):
    """Commit local changes."""

    def get_name(self) -> str:
        return "commit"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.usage = (
            'openxlab dataset commit [OPTIONS]\n\n'
            "Commit local changes.\n\n"
            'Example:\n'
            '> openxlab dataset commit --dataset-repo "username/dataset-repo-name" --commit-message "message"\n'
        )
        parser.add_argument(
            "-r",
            "--dataset-repo",
            type=str,
            required=True,
            help='The address of the dataset repository to commit.format: username/dataset-repo-name.[required]',
        )
        parser.add_argument(
            "-m",
            "--commit-message",
            type=str,
            required=True,
            help='The detail message to commit of the repository.[required]',
        )

    def take_action(self, parsed_args: Namespace) -> int:
        commit(dataset_repo=parsed_args.dataset_repo, commit_message=parsed_args.commit_message)

        return 0
