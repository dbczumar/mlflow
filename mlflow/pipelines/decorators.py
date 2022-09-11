_MLP_CODE_PATH = None

def _set_mlp_code_path(path):
    global _MLP_CODE_PATH

    _MLP_CODE_PATH = path


def mlp(fn):
    if _MLP_CODE_PATH is None:
        # Check if in notebook. If so, warn user that they probably want to be
        # running this in a cell with magic %%mlp_code
        return fn 

    # Using https://stackoverflow.com/questions/67631/how-do-i-import-a-module-given-the-full-path instead and having
    # step implementations import the module from the path is probably better
    mlp_fn_name = _MLP_CODE_PATH.rstrip(".py").replace("/", ".")
    mlp_fn_name = mlp_fn_name + f".{fn.__name__}" 

    setattr(fn, "mlp_fn_name", mlp_fn_name)

    return fn
