"""The two failure modes the HTTP layer has to distinguish, kept free of heavy imports.

`server/app.py` imports these directly so the request-validation path can be tested, and the
error contract read, without torch present.
"""


class OptionBudgetError(ValueError):
    """A question has more options than `head_max_len` can hold markers for. Answered with 422."""


class OverloadedError(RuntimeError):
    """More question rows in flight than `LAYA_MAX_QUEUE` allows. Answered with 529."""
