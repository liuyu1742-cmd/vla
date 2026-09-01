from concurrent.futures import ThreadPoolExecutor


class BizThreadPool(object):
    def __init__(self, max_workers=5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit_task(self, task, *args, **kwargs):
        future = self.executor.submit(task, *args, **kwargs)
        return future


# bury data thread pool
bury_thread_pool = BizThreadPool(5)
