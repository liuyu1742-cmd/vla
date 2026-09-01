"""
model inference-cli
"""

from argparse import ArgumentParser
from argparse import Namespace

from openxlab.model.handler.model_inference import Inference as ModelInference
from openxlab.types.command_type import BaseCommand


class Inference(BaseCommand):
    """inference"""

    def get_name(self) -> str:
        return "inference"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "-r",
            "--model-repo",
            required=True,
            help="model repository address. format:username/repository.",
        )
        parser.add_argument(
            "-i", "--input", type=str, nargs="+", required=True, help="input content, text."
        )
        parser.add_argument("-n", "--model-name", required=True, help="model name.")
        parser.add_argument("-d", "--device", required=False, help="device name.")

    def take_action(self, parsed_args: Namespace) -> int:
        inferencer = ModelInference(
            parsed_args.model_repo, parsed_args.model_name, parsed_args.device
        )
        inferencer.inference(parsed_args.input)
        return 0
