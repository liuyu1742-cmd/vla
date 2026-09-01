"""
get dataset repository totally
"""

import os
import sys
import platform
import logging
from typing import Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from rich import print as rprint

# 检测操作系统
system_type = platform.system()
# MAC系统特殊处理标志
IS_MAC = system_type == 'Darwin'

from openxlab.dataset.commands.utility import ContextInfoNoLogin
from openxlab.dataset.constants import FILE_THRESHOLD
from openxlab.dataset.io import downloader
from openxlab.dataset.utils import bytes2human
from openxlab.dataset.utils import calculate_file_sha256
from openxlab.dataset.utils import format_progress_string
from openxlab.xlab.handler.user_token import trigger_update_check


def get(dataset_repo: str, target_path=""):
    """
    Get the dataset repository.

    Example:
        openxlab.dataset.get(
            dataset_repo="username/dataset_repo_name",
            target_path="/path/to/local/folder"
        )

    Parameters:
        @dataset_repo String The address of dataset repository.
        @target_path String The target local path to save the dataset repository.
    """
    # 日志：开始执行get命令
    logger.info(f"开始执行get命令，数据集仓库: {dataset_repo}")
    print(f"开始执行get命令，数据集仓库: {dataset_repo}")
    
    # update check
    trigger_update_check()

    if not target_path:
        target_path = os.getcwd()
    if target_path.startswith('~'):
        target_path = os.path.expanduser(target_path)
    target_path = os.path.realpath(target_path)
    
    # 日志：目标保存路径
    logger.info(f"数据集将保存到: {target_path}")
    print(f"数据集将保存到: {target_path}")

    ctx = ContextInfoNoLogin()
    client = ctx.get_client()

    # parse dataset_name
    parsed_ds_name = dataset_repo.replace("/", ",")
    parsed_save_path = dataset_repo.replace("/", "___")

    rprint("Fetching the list of datasets...")
    logger.info("开始获取数据集文件列表")
    
    # payload
    get_payload = {}
    
    # Pagination parameters with cursor-based approach
    after = None
    limit = 500
    all_files = []
    has_more = True
    
    # Get all pages of files using cursor pagination
    info_dataset_id = None
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
    client.get_api().track_download_dataset_files(dataset_name=parsed_ds_name, file_path="")
    # 日志：下载完成
    logger.info(f"数据集下载完成，保存路径: {local_file_path}")
    if IS_MAC:
        print("Download Completed.")
        print(f"The {obj} has been successfully downloaded to {local_file_path}")
        sys.stdout.flush()
    else:
        rprint("Download Completed.")
        rprint(f"The {obj} has been successfully downloaded to {local_file_path}")


def process_download_files(
    client, object_info_list, target_path, parsed_save_path, info_dataset_id
) -> Tuple[str, str]:
    # obtain num of files to download
    total_files = len(object_info_list)
    total_size = sum(file['size'] for file in object_info_list)
    finished_size = 0

    # 日志：开始下载文件
    logger.info(f"开始下载 {total_files} 个文件，总大小: {bytes2human(total_size,format='%(value).2f%(symbol)s')}")
    rprint(f"Downloading {len(object_info_list)} files: ")

    for idx in range(len(object_info_list)):
        file_size = object_info_list[idx]['size']
        file_name = object_info_list[idx]['name']
        file_path = os.path.join(target_path, parsed_save_path, file_name)
        
        # 日志：处理第(idx+1)个文件
        logger.info(f"处理第 {idx+1}/{total_files} 个文件: {file_name}, 大小: {bytes2human(file_size,format='%(value).2f%(symbol)s')}")

        # update downloaded files size and progress
        finished_size += file_size
        # Handle division by zero if total_size is 0
        progress = 100 if total_size == 0 else round((finished_size / total_size) * 100)
        msg = format_progress_string(progress, idx, total_files, finished_size, total_size)

        # file exist already
        if os.path.exists(file_path):
            # 日志：文件已存在，检查SHA256
            logger.info(f"文件已存在: {file_path}，检查SHA256是否匹配")
            # calculate file sha256
            file_sha256 = calculate_file_sha256(file_path=file_path)
            if file_sha256 == object_info_list[idx]['sha256']:
                if idx >= 1:
                    # clear msg in terminal of total progress
                    if IS_MAC:
                        # MAC系统上使用更简单的回车方式
                        print("\r", end="")
                    else:
                        print("\033[2K\r", end="")
                    sys.stdout.flush()
                rprint(f"{idx+1}. {file_path} already exists, jumping to next!")
                rprint(msg, end="")

                # the final msg of total progress
                if idx + 1 == total_files:
                    if IS_MAC:
                        print("\r", end="")
                    else:
                        print("\033[2K\r", end="")
                    sys.stdout.flush()
                    rprint(msg)
                    break

                continue

        # big file download
        if file_size > FILE_THRESHOLD:
            # 日志：大文件下载
            logger.info(f"开始大文件下载: {file_name}, 大小: {bytes2human(file_size,format='%(value).2f%(symbol)s')}")
            # add a new line to print progress of big file
            sys.stdout.write("\n")
            sys.stdout.flush()
            logger.info(f"获取下载URL: {file_name}")
            download_url = client.get_api().get_dataset_download_urls(
                info_dataset_id, object_info_list[idx]
            )
            downloader.BigFileDownloader(
                url=download_url,
                filename=file_name,
                idx=idx,
                download_dir=os.path.join(target_path, parsed_save_path),
                file_size=file_size,
                blocks_num=8,
            ).start()
            # clear msgs before two lines in terminal of total progress and progress of big file
            if IS_MAC:
                # MAC系统上使用更简单的清除方式
                print("\n\n\r", end="")
            else:
                print("\033[1A\033[2K\033[1B\033[2K\033[1A\r", end="")
            sys.stdout.flush()

        # small file download
        else:
            # 日志：小文件下载
            logger.info(f"开始小文件下载: {file_name}, 大小: {bytes2human(file_size,format='%(value).2f%(symbol)s')}")
            logger.info(f"获取下载URL: {file_name}")
            download_url = client.get_api().get_dataset_download_urls(
                info_dataset_id, object_info_list[idx]
            )
            downloader.SmallFileDownload(
                url=download_url,
                filename=file_name,
                download_dir=os.path.join(target_path, parsed_save_path),
            )._single_thread_download()

            # clear the output of total downloaded progress in terminal if file idx != 1
            if idx >= 1:
                if IS_MAC:
                    print("\r", end="")
                else:
                    print("\033[2K\r", end="")
                sys.stdout.flush()

        # print progress msg of every new downloaded files
        if IS_MAC:
            # MAC系统上使用普通print代替rich.print，确保兼容性
            print(f"{idx+1}. file: {file_name}, size: {bytes2human(file_size,format='%(value).2f%(symbol)s')}, progress: 100%")
            sys.stdout.flush()
        else:
            rprint(
                f"{idx+1}. file: {file_name}, size: {bytes2human(file_size,format='%(value).2f%(symbol)s')}, progress: 100%"
            )

        # the final new download file needs a new line
        if idx + 1 == total_files:
            # print with "\n"
            if IS_MAC:
                print(msg)
                sys.stdout.flush()
            else:
                rprint(msg)
        else:
            # print without "\n"
            if IS_MAC:
                print(msg, end="")
                sys.stdout.flush()
            else:
                rprint(msg, end="")
    # 日志：所有文件处理完成
    logger.info(f"所有 {total_files} 个文件处理完成，总大小: {bytes2human(total_size,format='%(value).2f%(symbol)s')}")
    
    # obtain the download path of file or folder
    if len(object_info_list) == 1:
        return 'file', file_path
    else:
        return 'folder', os.path.dirname(file_path)
