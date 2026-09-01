"""
download specific file/files according to source_path(single file/relative path) of dataset repository
"""
import os
import platform
import re
import sys
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from rich import print as rprint

# 检测操作系统
system_type = platform.system()
# MAC系统特殊处理标志
IS_MAC = system_type == 'Darwin'

from openxlab.dataset.commands.utility import ContextInfoNoLogin
from openxlab.dataset.handler.get_dataset_repository import process_download_files
from openxlab.xlab.handler.user_token import trigger_update_check


def download(dataset_repo: str, source_path: str, target_path=""):
    """
    Download file or folder of a dataset repository.

    Example:
        openxlab.dataset.download(
            dataset_repo="username/dataset_repo_name",
            source_path="/raw/file",
            target_path="/path/to/local/folder"
        )

    Parameters:
        @dataset_repo String The address of dataset repository.
        @source_path String The relative path of the target file or folder to download.
        @target_path String The target local path to store the file or folder.
    """
    # 日志：开始执行download命令
    logger.info(f"开始执行download命令，数据集仓库: {dataset_repo}，源路径: {source_path}")
    print(f"开始执行download命令，数据集仓库: {dataset_repo}，源路径: {source_path}")
    
    # update check
    trigger_update_check()

    if not target_path:
        target_path = os.getcwd()
    if target_path.startswith('~'):
        target_path = os.path.expanduser(target_path)
    target_path = os.path.realpath(target_path)
    
    # 日志：目标保存路径
    logger.info(f"文件将保存到: {target_path}")
    print(f"文件将保存到: {target_path}")

    # remove prefix of . in soure_path
    source_path = re.sub(r'^\.+', '', source_path)
    
    # 日志：处理后的源路径
    logger.info(f"处理后的源路径: {source_path}")

    ctx = ContextInfoNoLogin()
    client = ctx.get_client()

    # parse dataset_name
    parsed_ds_name = dataset_repo.replace("/", ",")
    # huggingface use underscores when loading/downloading datasets
    parsed_save_path = dataset_repo.replace("/", "___")

    rprint("Fetching the list of files...")
    logger.info("开始获取文件列表")
    get_payload = {"prefix": source_path}
    
    # Pagination parameters with cursor-based approach
    after = None
    limit = 500
    all_files = []
    has_more = True
    info_dataset_id = None
    
    # Get all pages of files using cursor pagination
    while has_more:
        # 日志：获取文件列表
        logger.info(f"正在获取文件列表，cursor: {after}")
        data_dict = client.get_api().get_dataset_files(
            dataset_name=parsed_ds_name, 
            payload=get_payload, 
            needContent=True,
            after=after,
            limit=limit
        )
        
        # Store dataset_id from the first page
        if after is None and data_dict['list']:
            info_dataset_id = data_dict['list'][0]['dataset_id']
        
        # Add current page files to all_files list
        current_files = data_dict['list']
        all_files.extend(current_files)
        
        # Check if there are more pages using hasNext flag
        has_more = data_dict.get('hasNext', False)
        # Update after cursor for next page if available
        if has_more:
            after = data_dict.get('after')
    
    object_info_list = []
    for info in all_files:
        curr_dict = {}
        curr_dict['size'] = info['size']
        curr_dict['name'] = info['path'][1:]
        # Use get method to safely access sha256 field with default value
        curr_dict['sha256'] = info.get('sha256', '')
        # without destination path upload file,file has prefix with '//'
        if info['path'].startswith('//'):
            curr_dict['name'] = info['path'][2:]
        object_info_list.append(curr_dict)

    if object_info_list:
        # 日志：获取文件列表完成
        logger.info(f"文件列表获取完成，共 {len(object_info_list)} 个文件")
        print(f"文件列表获取完成，共 {len(object_info_list)} 个文件")
        
        # download check for crawler with one file
        download_check_path = object_info_list[0]['name']
        # download check
        logger.info(f"执行下载前检查，文件路径: {download_check_path}")
        client.get_api().download_check(dataset_id=info_dataset_id, path=download_check_path)
        # 日志：开始处理下载文件
        logger.info(f"开始处理下载文件，目标路径: {os.path.join(target_path, parsed_save_path)}")
        obj, local_file_path = process_download_files(
            client, object_info_list, target_path, parsed_save_path, info_dataset_id
        )

    client.get_api().track_download_dataset_files(
        dataset_name=parsed_ds_name, file_path=source_path
    )
    # 日志：下载完成
    logger.info(f"文件下载完成，保存路径: {local_file_path}")
    if IS_MAC:
        print("Download Completed.")
        print(f"The {obj} has been successfully downloaded to {local_file_path}")
        sys.stdout.flush()
    else:
        rprint("Download Completed.")
        rprint(f"The {obj} has been successfully downloaded to {local_file_path}")
