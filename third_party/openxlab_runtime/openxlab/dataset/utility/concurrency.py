from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
import os
import platform
import threading
from threading import Event
import time
from typing import Callable
from typing import Iterable


# keyboard interrupted or exception
error_event = Event()
complete_event = Event()


def wait_result(futures: list, results: list):
    try:
        for future in as_completed(futures):
            results.append(future.result())
        complete_event.set()
    except Exception:
        error_event.set()
        raise


def concurrent_submit(func: Callable, workers: int, *args):
    def wrapper(*args):
        try:
            return func(*args)
        except Exception as e:
            error_event.set()
            raise Exception(f"An error occurred. Error: {e}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(wrapper, *args) for _ in range(workers)]
        results = []
        threading.Thread(target=wait_result, args=[futures, results], daemon=True).start()

        while not (complete_event.is_set() or error_event.is_set()):
            time.sleep(1)

        # if error_event.is_set():
        #     raise Exception("Keyboard interrupted.")

        return results


def concurrent_map(func: Callable, workers: int, iters: Iterable):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(func, iters)
        return results


def is_mac():
    return platform.system() == "Darwin"


def init_worker_num(part_num: int):
    # if is_mac():
    #     return 1
    worker = int(os.cpu_count() / 4)
    worker = max(worker, 1)
    return worker


if __name__ == "__main__":
    print("+++++++++++++++ test ++++++++++++++++++")
    t_list = [1, 2, 3, 4, 5]
    concurrent_map(lambda x: print(x + 1), 5, t_list)
    print(t_list)
