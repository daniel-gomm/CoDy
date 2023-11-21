import logging
from io import StringIO

from tqdm import tqdm, tqdm_notebook
import IPython


def construct_model_path(path_prefix: str, model_name: str, data_name: str, epoch: str = None):
    if epoch:
        return f'{path_prefix}{model_name}-{data_name}-{epoch}.pth'
    return f'{path_prefix}{model_name}-{data_name}.pth'


def _is_running_in_notebook() -> bool:
    try:
        shell = IPython.get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False
    except NameError:
        return False


class ProgressBar:

    def __init__(self, max_item: int, prefix: str = ''):
        self.running_in_notebook = _is_running_in_notebook()
        if self.running_in_notebook:
            self.progress_bar = tqdm_notebook(total=max_item, desc=prefix)
        else:
            self.desc = tqdm(total=0, position=1, bar_format='{desc}')
            self.progress_bar = tqdm(total=max_item, desc=prefix, dynamic_ncols=True, position=0)
        self.inner_progress = None
        self.inner_current_value = 0
        self.current_value = 0
        self.log_stream = StringIO()
        logging.basicConfig(stream=self.log_stream, level=logging.INFO)
        self.logger = logging.getLogger()

    def next(self):
        self.current_value += 1
        self.progress_bar.update(1)
        self.progress_bar.refresh()

    def reset(self, total: int = 100):
        self.current_value = 0
        self.progress_bar.reset(total=total)

    def close(self):
        self.progress_bar.close()

    def update_postfix(self, postfix: str):
        self.progress_bar.postfix = postfix
        self.progress_bar.update(0)

    def add_inner_progress(self, max_item: int, prefix: str):
        assert self.inner_progress is None, 'Inner progress bar already exists'
        self.inner_current_value = 0
        if self.running_in_notebook:
            self.inner_progress = tqdm_notebook(total=max_item, desc=prefix)
        else:
            self.inner_progress = tqdm(total=max_item, desc=prefix, position=2, leave=False, dynamic_ncols=True)

    def inner_next(self):
        self.inner_current_value += 1
        self.inner_progress.update(1)

    def inner_close(self):
        self.inner_progress.clear()
        self.inner_progress.close()
        self.inner_progress = None

    def write(self, message: str, log: bool = True):
        if log:
            self.logger.info(message)
            msg = self.log_stream.getvalue()
            self.log_stream.flush()
        else:
            msg = message
        if self.running_in_notebook:
            print(msg)
            return
        self.desc.clear()
        self.desc.set_description_str(msg)
        self.progress_bar.update(0)
