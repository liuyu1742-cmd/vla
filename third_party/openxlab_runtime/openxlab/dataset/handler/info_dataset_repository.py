"""
query to get the information of dataset repository
"""
from rich import box
from rich.console import Console
from rich.table import Table

from openxlab.dataset.commands.utility import ContextInfoNoLogin
from openxlab.dataset.constants import MAX_FILES_PER_DIR
from openxlab.dataset.utils import bytes2human
from openxlab.xlab.handler.user_token import trigger_update_check


def info(dataset_repo: str):
    """
    Get the information of a dataset repository.
    Note: if you are not log in, you can only get the information of public dataset repository.

    Example:
        openxlab.dataset.info(dataset_repo="username/dataset_repo_name")

    Parameters:
        @dataset_repo String The address of dataset repository.
    """
    # update check
    trigger_update_check()

    ctx = ContextInfoNoLogin()
    client = ctx.get_client()

    parsed_ds_name = dataset_repo.replace("/", ",")

    info_data = client.get_api().get_dataset_info(dataset_name=parsed_ds_name)
    # get file list
    data_dict = client.get_api().get_dataset_files(
        dataset_name=parsed_ds_name, needContent=True, auth=False
    )

    file_path_dict = {}
    total_size = 0
    for file in data_dict['list']:
        file_path_dict[file['path']] = bytes2human(file['size'], format='%(value).2f%(symbol)s')
        total_size += file['size']
    total_size = bytes2human(total_size)
    file_paths = filter_paths(file_path_dict=file_path_dict, max_files_per_dir=MAX_FILES_PER_DIR)

    info_data_result = _reformat_info_data(info_data)
    # create and print table in console
    inner_table = create_file_list_table(file_paths, total_size=total_size)

    console = Console()
    table = Table(show_header=True, header_style='bold cyan', box=box.ASCII2)
    table.add_column("Field", width=20, justify='full', overflow='fold')
    table.add_column("Content", width=120, justify='full', overflow='fold')

    for key in info_data_result.keys():
        val = info_data_result[key]
        val = "" if not val else val
        table.add_row(key, val, end_section=True)
    table.add_row('File List', inner_table)
    console.print(table)

    # update info data with file list for info SDK func
    file_list_dict = {'Total size': total_size}
    file_list_dict.update(file_paths)
    info_data_result['File List'] = file_list_dict

    return info_data_result


def create_file_list_table(file_paths: dict, total_size):
    """Create the table for the key of 'File List'"""
    # table for File List
    inner_table0 = Table(show_header=True, header_style="cyan", border_style="", box=box.ASCII2)
    inner_table0.add_column("File", justify="left")
    inner_table0.add_column("Size", justify="right")

    for dir_name, file_dict in file_paths.items():
        inner_table0.add_row(f'- {dir_name}', end_section=True)
        for file_path, file_size in file_dict.items():
            inner_table0.add_row(f'   - {file_path}', file_size, end_section=True)

    inner_table1 = Table(show_header=False, show_edge=False)
    inner_table1.add_row(f"Total Size: {total_size}")
    inner_table1.add_row(inner_table0)

    return inner_table1


def filter_paths(file_path_dict, max_files_per_dir=MAX_FILES_PER_DIR):
    """Filter file paths and limit the number of files displayed per directory.

    Args:
        file_path_dict (dict): A dictionary containing file paths as keys and their sizes as values.
        max_files_per_dir (int, optional): Maximum number of files to display per directory.
        Defaults to MAX_FILES_PER_DIR.

    Returns:
        dict: A dictionary containing filtered file paths and sizes.
    """
    file_paths = {'/': {}}
    root_dirs = {}

    for path, size in file_path_dict.items():
        parts = path.strip('/').split('/', 1)

        # handle files in the root directory (no '/' or only one '/')
        if len(parts) == 1:
            file_paths['/'][parts[0]] = size
        # handle second-level directories
        elif len(parts) == 2:
            dir_name, file_name = parts
            dir_with_prefix = f'/{dir_name}/'
            if dir_with_prefix not in root_dirs:
                root_dirs[dir_with_prefix] = {}

            root_dirs[dir_with_prefix][file_name] = size

    # Process second-level directories and limit the number of files
    for dir_name, files in root_dirs.items():
        # sort files by file name
        sorted_files = dict(sorted(files.items()))

        # take the first max_files_per_dir files
        file_paths[dir_name] = dict(list(sorted_files.items())[:max_files_per_dir])

        # if the number of files exceeds the limit, add an ellipsis entry
        if len(files) > max_files_per_dir:
            file_paths[dir_name][
                f'...\n  (Showing {max_files_per_dir} of {len(files)} files)'
            ] = ''

    return file_paths


def _format_types(info_data, type_name):
    types_str = ""
    if type_name in info_data['attrs'].keys():
        types_list = info_data['attrs'][type_name]
        if types_list and len(types_list) > 0:
            types_str = ", ".join([x['name']['en'] for x in types_list])

    return types_str


def _reformat_info_data(info_data):
    publisher_str = _format_types(info_data, 'publisher')
    media_types_str = _format_types(info_data, 'mediaTypes')
    label_types_str = _format_types(info_data, 'labelTypes')
    task_types_str = _format_types(info_data, 'taskTypes')
    if info_data['introduction']:
        data_introduction = info_data['introduction']['en']
    else:
        data_introduction = ""
    introduction_str = ""
    if data_introduction and len(data_introduction) > 0:
        introduction_str = data_introduction[:97] + '...'

    info_data_result = {
        'Name': info_data['name'],
        'Introduction': introduction_str,
        'Author': publisher_str,
        'Data Type': media_types_str,
        'Label Type': label_types_str,
        'Task Type': task_types_str,
    }

    return info_data_result
