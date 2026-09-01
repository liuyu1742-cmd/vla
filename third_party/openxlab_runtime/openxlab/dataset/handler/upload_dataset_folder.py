"""
upload local folder to dataset repository
"""

from openxlab.dataset.commands.utility import ContextInfoNoLogin
from openxlab.dataset.io.upload import Uploader
from openxlab.xlab.handler.user_token import trigger_update_check


def upload_folder(dataset_repo: str, source_path: str, target_path="", upload_network="cdn"):
    """
    Upload folder from local to remote.

    Example:
        openxlab.dataset.upload_folder(
            dataset_repo="username/dataset_repo_name",
            source_path="/path/to/local/folder",
            target_path="/raw/folder",
            upload_network="vpc"
        )

    Parameters:
        @dataset_repo String The address of dataset repository.
        @source_path String The local path of the folder to upload.
        @target_path String The target path to upload folder.
        @upload_network String Specifies the network type for file upload,
        cdn (default): Use Content Delivery Network for faster distribution.
        vpc: Use Virtual Private Cloud for enhanced security and privacy.

    """
    try:
        # update check
        trigger_update_check()

        ctx = ContextInfoNoLogin()
        client = ctx.get_client().get_api()
        parsed_ds_name = dataset_repo.replace("/", ",")
        uploader = Uploader(client, dataset_repo_name=parsed_ds_name)
        uploader.upload_folder(source_path, target_path, upload_network=upload_network)
    except Exception as e:
        print(f"{e}")
