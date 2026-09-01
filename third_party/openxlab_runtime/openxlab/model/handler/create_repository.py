"""
创建模型仓库
"""

import os
import re

from openxlab.model.clients.openapi_client import OpenapiClient
from openxlab.model.common.bury import bury_data
from openxlab.model.common.constants import endpoint
from openxlab.model.common.constants import token
from openxlab.model.common.meta_file_util import MetafileParser
from openxlab.model.handler.upload_file import upload_model_list


@bury_data("create_model")
def create(model_repo, source="metafile.yaml", private=False) -> None:
    """
    create model repository
    usage: cli & sdk
    """
    try:
        # split params
        username, repository = _split_repo(model_repo)
        client = OpenapiClient(endpoint, token)
        # parse & check metafile.yml
        print("Current directory:", os.getcwd())

        meta_parser = MetafileParser(source)
        meta_data = meta_parser.parse_and_validate()
        client.create_repository(repository, private, meta_data)
        print(f"repository:{repository} created successfully.")
        # upload file
        meta_file_model_list = meta_data["Models"]
        remote_model_list = [
            {**d, "name": d["Name"], "weightName": os.path.basename(d["Weights"])}
            for d in meta_file_model_list
        ]
        upload_model_list(
            client, repository, meta_file_model_list, remote_model_list, domain="bucket"
        )
        print("file upload successfully.")
    except ValueError as e:
        print(f"Error: {e}")
        return


def _split_repo(model_repo) -> (str, str):
    """
    Split a full repository name into two separate strings: the username and the repository name.
    """
    # username/repository format check
    pattern = r".+/.+"
    if not re.match(pattern, model_repo):
        raise ValueError("The input string must be in the format 'username/model_repo'")

    values = model_repo.split("/")
    return values[0], values[1]


def _parse_check():
    pass
