import io
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import threading
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from urllib.parse import urljoin

from filelock import FileLock
import requests
from requests import Response
from tqdm import tqdm

from openxlab.dataset.constants import BASE_DELAY
from openxlab.dataset.constants import CKPT_FOLDER
from openxlab.dataset.constants import CKPT_SUFFIX
from openxlab.dataset.constants import MAX_DELAY
from openxlab.dataset.constants import MAX_RETRIES
from openxlab.dataset.constants import MIN_FILE_SEGMENTATION
from openxlab.dataset.constants import README_FILE_NAME
from openxlab.dataset.constants import UPLOAD_TIMEOUT
from openxlab.dataset.utility.concurrency import concurrent_submit
from openxlab.dataset.utility.concurrency import error_event
from openxlab.dataset.utility.concurrency import init_worker_num
from openxlab.dataset.utils import calculate_file_sha256
from openxlab.dataset.utils import get_file_content
from openxlab.dataset.utils import retry_with_backoff


lock = threading.Lock()


def delete_file(file_path):
    if file_path is None:
        pass
    if os.path.exists(file_path):
        lock_file_path = file_path + ".lock"
        f_lock = FileLock(lock_file=lock_file_path)
        with f_lock:
            if os.path.isfile(file_path):
                os.remove(file_path)


def signal_handler(sig, frame):
    # 设置停止事件
    error_event.set()


signal.signal(signal.SIGINT, signal_handler)


class FileInfo:
    def __init__(
        self,
        client,
        abs_path: str,
        rel_path: Optional[str] = None,
        upload_network: Optional[str] = "cdn",
    ):
        self.abs_path = abs_path
        self.rel_path = rel_path
        self.sha256: Optional[str] = None
        self.size: Optional[int] = None
        self.upload_process: Optional[List[Dict]] = []
        self.dataset: Optional[str] = None
        self.branch: Optional[str] = None
        self.sink_path: Optional[str] = None
        self.upload_id: Optional[str] = None
        self.client = client
        self.upload_network = upload_network
        self.total_parts_count: Optional[int] = None
        self.part_size: Optional[int] = None
        self.uploaded_parts: Optional[List[int]] = []
        self.ckpt_file_path: Optional[str] = None
        self.ckpt_lock_file: Optional[str] = None
        self.post_upload_in_progress = False

    def to_dict(self):
        return {
            "abs_path": self.abs_path,
            "rel_path": self.rel_path,
            "sha256": self.sha256,
            "size": self.size,
            "dataset": self.dataset,
            "branch": self.branch,
            "sink_path": self.sink_path,
            "upload_id": self.upload_id,
            "total_parts_count": self.total_parts_count,
            "part_size": self.part_size,
            "uploaded_parts": self.uploaded_parts,
            "upload_process": self.upload_process,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, indent=4)

    def update_upload_info(self, part_num: int, part_info: dict):
        with lock:
            self.uploaded_parts.append(part_num)
            self.upload_process.append(part_info)

    def get_size(self):
        self.size = os.stat(self.abs_path).st_size

    def store_ckpt_to_local(self):
        self.ckpt_lock_file = self.ckpt_file_path + ".lock"
        f_lock = FileLock(lock_file=self.ckpt_lock_file)
        with f_lock:
            with open(self.ckpt_file_path, "w", encoding="utf-8") as f:
                f.write(self.to_json())

    def get_pre_upload_info(self, dataset, branch, sink_path: Optional[str] = None):
        self.dataset = dataset
        self.branch = branch

        if self.size is None:
            self.get_size()
        if self.sha256 is None:
            self.sha256 = calculate_file_sha256(self.abs_path)
        if self.rel_path is None:
            self.rel_path = os.path.split(self.abs_path)[-1]

        if self.sink_path is None:
            self.sink_path = self.rel_path if sink_path is None else sink_path

        # check ckpt file
        file_dir = os.path.dirname(self.abs_path)
        if self.size > MIN_FILE_SEGMENTATION:
            # optimize: can change the ckpt file path
            self.ckpt_file_path = os.path.join(file_dir, CKPT_FOLDER, self.rel_path + CKPT_SUFFIX)
            self.get_uploaded_parts_info()
        req_dict = {
            "size": self.size,
            "sha256": self.sha256,
            "network": self.upload_network,
            "id": self.upload_id,
        }
        self.pre_upload = self.client.pre_object_upload(
            self.dataset, self.branch, self.sink_path, req_dict
        )
        # retry timeout or an unknown error, need to clear the cache.
        if self.upload_id is not None and self.upload_id != self.pre_upload["id"]:
            delete_file(self.ckpt_file_path)
            self.uploaded_parts = []
            self.upload_process = []

        if not self.pre_upload["exists"]:
            self.upload_id = self.pre_upload["id"]
            self.total_parts_count = len(self.pre_upload["parts"])
            self.part_size = self.pre_upload["partSize"]

    def get_uploaded_parts_info(self):
        if os.path.exists(self.ckpt_file_path):
            with lock:
                with open(self.ckpt_file_path, "r") as f:
                    content = f.read()
                    ckpt_info = json.loads(content)
                self.uploaded_parts = ckpt_info["uploaded_parts"]
                self.upload_process = ckpt_info["upload_process"]
                self.upload_id = ckpt_info["upload_id"]

    def put_part(self, part_number: int):
        try:
            if self.upload_id is None:
                raise Exception("no upload id info")

            part_number_list = [part["number"] for part in self.pre_upload["parts"]]
            if part_number not in part_number_list:
                raise Exception("the part number does not exist")

            part = [part for part in self.pre_upload["parts"] if part["number"] == part_number][0]
            part_put_url = part["putUrl"]
            part_size = self.pre_upload["partSize"]

            # init put_resp
            put_resp = Response()
            put_resp.status_code = -1
            if not error_event.is_set():
                if part["number"] == 0:
                    with Path(self.abs_path).open("rb") as data:
                        put_resp = put_file(url=part_put_url, data=data)
                else:
                    offset = (part_number - 1) * part_size
                    read_size = min(self.size - offset, part_size)
                    with lock:
                        with open(self.abs_path, "rb") as f:
                            f.seek(offset)
                            with io.BufferedReader(io.BytesIO(f.read(read_size))) as data:
                                put_resp = put_file(url=part_put_url, data=data)

            if put_resp is None:
                raise Exception("failed to put file to server.")
            if put_resp.status_code != 200:
                raise Exception(
                    f"put failed, status_code = {put_resp.status_code}, text = {put_resp.text}"
                )
            etag = put_resp.headers["Etag"]
            part_submit_dict = {"number": part_number, "etag": etag}

            self.update_upload_info(part_num=part_number, part_info=part_submit_dict)
            if not error_event.is_set() and self.total_parts_count > 1:
                with lock:
                    ckpt_dir = os.path.dirname(self.ckpt_file_path)
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir, exist_ok=True)
                    self.store_ckpt_to_local()
        except Exception as e:
            raise RuntimeError(f"put file failed. Error: {e}")

    def get_post_upload_info(self):
        if not self.upload_process or len(self.upload_process) != len(self.pre_upload["parts"]):
            raise Exception("no upload task or upload does not finish")
        post_req = {"id": self.upload_id, "parts": self.upload_process}
        try:
            post_resp = self.client.post_object_upload(
                self.dataset, self.branch, self.sink_path, post_req
            )
            if post_resp["sha256"] != self.sha256:
                raise Exception(f"file with absolute path {self.abs_path} sha256 mismatch!")
        except Exception:
            raise
        finally:
            if self.ckpt_file_path:
                delete_file(self.ckpt_file_path)
            if self.ckpt_lock_file:
                delete_file(self.ckpt_lock_file)


task_list: List[Tuple[FileInfo, int, int]] = []  # 全局任务列表


@retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
def put_file(url: str, data):
    if not error_event.is_set():
        put_resp = requests.put(url=url, data=data, timeout=UPLOAD_TIMEOUT)
        return put_resp
    return None


def is_hidden(filepath):
    """
    Check if the file or directory is hidden.
    This works on both Windows and Unix-like systems.
    """
    # Unix-like systems
    if filepath.startswith("."):
        return True
    # Windows systems
    if os.name == "nt":
        import ctypes

        ATTR_HIDDEN = 0x02
        file_attr = ctypes.windll.kernel32.GetFileAttributesW(filepath)
        return bool(file_attr & ATTR_HIDDEN)
    return False


def _get_files_in_folder(
    client, folder_path: str, upload_network: str, mts: bool = False
) -> List[FileInfo]:
    if not Path(folder_path).is_dir():
        raise Exception(f"path {folder_path} is not a dir or does not exist")
    root_path = os.path.abspath(folder_path)
    res_list = multiprocessing.Manager().list() if mts else []

    for root, dirs, files in os.walk(root_path):
        # Filter out hidden directories from dirs list (this will prevent os.walk from entering them)
        dirs[:] = [d for d in dirs if not is_hidden(d)]
        # optimize: ignore all hidden files
        # files[:] = [f for f in files if not f.startswith('.')]
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(".DS_Store"):
                continue
            # add moderating README.md
            if README_FILE_NAME in file:
                res = moderate_readme(client=client, file_path=file_path)
                if not res:
                    print(
                        f"\033[5;33mWarning: {file} cannot be uploaded. Please confirm compliance and upload again!!! "
                        f"注意：{file} 文件不能上传，请确认文件内容合规后再次上传！！！\033[0m"
                    )
                    continue
            rel_path = os.path.relpath(file_path, root_path).replace("\\", "/")
            file_obj = FileInfo(client, file_path, rel_path, upload_network=upload_network)
            file_obj.get_size()
            res_list.append(file_obj)
    return res_list


def update_progress(update_value: int, progress: Optional[tqdm] = None):
    if progress:
        progress.update(update_value)


def upload_files_worker(
    task_list: List[Tuple[FileInfo, int, int]], progress: Optional[tqdm] = None
):
    while len(task_list) > 0 and not error_event.is_set():
        with lock:
            try:
                task = task_list.pop(0)
            except Exception:
                # task_list 为空，表示已完成
                return
        file = task[0]
        part_number = task[1]
        part_size = task[2]
        if file.upload_id is None and not error_event.is_set():
            update_progress(file.size, progress)
        else:
            if part_number not in file.uploaded_parts:
                file.put_part(part_number)
            if len(file.pre_upload["parts"]) == len(file.upload_process):
                with lock:
                    if file.post_upload_in_progress:
                        return
                    file.post_upload_in_progress = True
                    try:
                        file.get_post_upload_info()
                    except Exception as e:
                        raise Exception(f"post upload failed. Error:{e}")
            if not error_event.is_set():
                update_progress(part_size, progress)


def upload_files(
    task_list: List[Tuple[FileInfo, int, int]], progress: Optional[tqdm] = None, workers: int = 8
):
    concurrent_submit(upload_files_worker, workers, task_list, progress)


def files_to_tasks(
    file_list: List[FileInfo],
    task_list: List[Tuple[FileInfo, int, int]],
    dataset: str,
    branch: str,
    progress: Optional[tqdm] = None,
):
    while len(file_list) > 0 and not error_event.is_set():
        with lock:
            try:
                file = file_list.pop(0)
            except Exception:
                # file_list 为空，表示已遍历完成所有文件
                return
        file.get_pre_upload_info(dataset, branch)
        if file.upload_id:
            if len(file.pre_upload["parts"]) == 1:
                task = (file, 0, file.size)
                with lock:
                    task_list.append(task)
            else:
                for part in file.pre_upload["parts"]:
                    part_number = part["number"]
                    size = min(
                        file.pre_upload["partSize"],
                        file.size - (part_number - 1) * file.pre_upload["partSize"],
                    )
                    with lock:
                        task_list.append((file, part_number, size))
        else:
            task = (file, 0, 0)
            with lock:
                task_list.append(task)
        update_progress(1, progress)


def get_task_list(
    file_list: List[FileInfo],
    dataset: str,
    branch: str,
    progress: Optional[tqdm] = None,
    workers: int = 8,
):
    # task_list = []
    concurrent_submit(files_to_tasks, workers, file_list, task_list, dataset, branch, progress)
    return task_list


# 获取dataset id的方法，此处先做测试
def get_dataset_id(client, dataset_repo_name: str):
    dataset_id = str(client.get_dataset_info(dataset_repo_name)["id"])
    return dataset_id


class Uploader:
    def __init__(self, client, dataset_repo_name: str):
        self.dataset = dataset_repo_name
        self.branch = "main"  # TODO default to main this version
        self.buffer_size = 128 * 1024  # 128 kb
        self.workers = 1
        self.client = client

    def upload_folder(self, source_path: str, destination_path: str, upload_network: str):
        source_path = os.path.realpath(os.path.expanduser(source_path))

        if not destination_path:
            destination_path = "/"

        if not destination_path.endswith("/"):
            destination_path = destination_path + "/"
        print("Fetching files list...")
        global task_list, file_list
        file_list = _get_files_in_folder(self.client, source_path, upload_network=upload_network)
        for file in file_list:
            file.sink_path = urljoin(destination_path, file.rel_path)

        total_num = len(file_list)
        total_size = sum([file.size for file in file_list])

        with tqdm(total=total_num, ncols=100, desc="Preparing", unit_scale=True) as progress:
            self.workers = init_worker_num(part_num=total_num)
            task_list = get_task_list(file_list, self.dataset, self.branch, progress, self.workers)

        with tqdm(total=total_size, ncols=100, desc="Uploading", unit_scale=True) as progress:
            upload_workers = init_worker_num(len(task_list))
            upload_files(task_list, progress, upload_workers)
        # rprint("Upload folder successfully!")

    def upload_file(self, source_path: str, destination_path: str, upload_network: str):
        # parse ., .. and ~ in path to get the real path of file
        source_path = os.path.realpath(os.path.expanduser(source_path))

        if not os.path.isfile(source_path):
            raise Exception(f"{source_path} is not a file or does not exist!")
        if not destination_path:
            destination_path = "/"
        if not destination_path.endswith("/"):
            destination_path = destination_path + "/"

        file_obj = FileInfo(self.client, source_path, upload_network=upload_network)
        file_obj.get_size()
        file_obj.sink_path = urljoin(destination_path, os.path.split(source_path)[-1])
        print(urljoin(destination_path, os.path.split(source_path)[-1]))
        print("checking file info...")
        # moderate README.md
        if README_FILE_NAME in source_path:
            res = moderate_readme(client=self.client, file_path=source_path)
            if not res:
                sys.exit(-1)
        global task_list, file_list
        file_list = [file_obj]
        # task_list = []

        total_size = file_obj.size
        files_to_tasks(file_list, task_list, self.dataset, self.branch)
        with tqdm(total=total_size, ncols=100, desc="Uploading", unit_scale=True) as progress:
            upload_workers = init_worker_num(len(task_list))
            upload_files(task_list, progress, upload_workers)
        # rprint("Upload file successfully!")


def moderate_readme(client, file_path: str):
    # moderate README.md
    if not Path(file_path).is_file():
        raise Exception(f"file {file_path} does not exist")
    readme_content = get_file_content(file_path=file_path)
    moderate_readme_res = client.moderate_text(content=readme_content, text_type=3)
    return moderate_readme_res
