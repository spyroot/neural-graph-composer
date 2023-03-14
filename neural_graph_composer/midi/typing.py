from typing import Union, TypeVar, Tuple, Generator, List, Sequence

import numpy as np

CharLike_co = Union[str, bytes]
BoolLike = Union[bool, np.bool]
UIntLike = Union[BoolLike, "np.unsignedinteger[Any]"]
IntLike = Union[BoolLike, int, "np.integer[Any]"]
FloatLike = Union[IntLike, float, "np.floating[Any]"]
ComplexLike = Union[FloatLike, complex, "np.complexfloating[Any, Any]"]
TD64Like = Union[IntLike, np.timedelta64]

Generic_T = TypeVar("Generic_T")
IterableLike = Union[List[Generic_T], Tuple[Generic_T, ...], Generator[Generic_T, None, None]]
SequenceLike = Union[Sequence[Generic_T], np.ndarray]
ScalarOrSequence = Union[Generic_T, SequenceLike[Generic_T]]
