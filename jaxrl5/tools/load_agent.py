from typing import Any, Callable, Dict, Optional, Tuple

from jaxrl5.tools.load_ssm import load_ssm
from jaxrl5.tools.load_td3 import load_td3
from jaxrl5.tools.load_td3_lag import load_td3_lag


def load_agent(
    algo: str,
    ckpt_path: str,
    step: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[object, Callable, Dict]:
    algo = algo.lower()
    if algo == "td3":
        return load_td3(ckpt_path, step=step)
    if algo == "td3_lag":
        return load_td3_lag(ckpt_path, step=step)
    if algo == "ssm":
        return load_ssm(ckpt_path, step=step, **kwargs)
    raise ValueError(f"Unsupported algo '{algo}'. Expected one of: td3, td3_lag, ssm")
