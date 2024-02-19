import wandb
import functools

from ..types import Callback, CallbackPayload


def wandb_log_output(func: Callback, field_name: str, key: str, wandb_data_type, cleanup_func=None):
    """
    Wraps a function. Takes the return value and checks if it is a file. Files are synced to wandb.
    """
    @functools.wraps(func)
    def wrapper(payload: CallbackPayload):
        out = func(payload)
        if out is not None:
            wandb.log({
                f'{payload.split}/{payload.index}/{field_name}/{key}': wandb_data_type(out)
            }, commit=False)
        if cleanup_func is not None:
            cleanup_func(out)
    return wrapper
