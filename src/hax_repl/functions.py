from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from pydantic import BaseModel

TArgs = TypeVar("TArgs", bound=BaseModel)
TResult = TypeVar("TResult", bound=BaseModel)


@dataclass(frozen=True)
class FunctionSpec(Generic[TArgs, TResult]):
    name: str
    description: str
    args_model: type[TArgs]
    result_model: type[TResult]
    impl: Callable[[TArgs], TResult]
