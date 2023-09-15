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
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class ProgressBar:

    def __init__(self, max_item: int, prefix: str = ''):
        if _is_running_in_notebook():
            self.progress_bar = tqdm_notebook(total=max_item, desc=prefix)
        else:
            self.progress_bar = tqdm(total=max_item, desc=prefix)
        self.current_value = 0

    def next(self):
        self.current_value += 1
        self.progress_bar.update(1)

    def reset(self, total: int = 100):
        self.current_value = 0
        self.progress_bar.reset(total=total)

    def close(self):
        self.progress_bar.close()

    def update_postfix(self, postfix: str):
        self.progress_bar.postfix = postfix
