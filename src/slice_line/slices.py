from typing import Generator, TypeVar, Tuple, List
from pandas import DataFrame
import numpy as np
from numpy.typing import ArrayLike

SliceIdx = TypeVar("SliceIdx", ArrayLike, int)
Filtered = TypeVar("Filtered", ArrayLike, int)


def parse_predicates(
    in_names: ArrayLike, slices: SliceIdx
) -> Generator[SliceIdx, SliceIdx, Tuple[Filtered, Filtered]]:
    for s in slices:
        indices = np.where(s)
        yield zip(in_names[indices], s[indices])


def parse_query(
    predicates_gen: Generator[SliceIdx, SliceIdx, Tuple[Filtered, Filtered]]
) -> Generator[None, None, str]:
    slices = next(predicates_gen)
    while slices:
        try:
            prox = next(slices)
            query = ""
            while prox:
                name, value = prox
                if isinstance(value, str):
                    query += f'{name} == "{value}"'
                else:
                    query += f"{name} == {value}"
                try:
                    prox = next(slices)
                    query += " and "
                except StopIteration:
                    yield query
                    break
            slices = next(predicates_gen)
        except StopIteration:
            break


def get_slice_indices(
    dataset: DataFrame,
    query_parser: Generator[SliceIdx, SliceIdx, Tuple[Filtered, Filtered]],
) -> List:
    return [dataset.query(query).index.values for query in query_parser]


def efect_size(pred_error: ArrayLike, slice_idx: SliceIdx, idx: ArrayLike) -> float:
    mask = np.isin(idx, slice_idx)
    not_mask = ~mask
    slice_error = pred_error[mask]
    complement_error = pred_error[not_mask]
    error = np.mean(slice_error) - np.mean(complement_error)
    slice_var = np.var(slice_error)
    complement_var = np.var(complement_error)
    std = np.sqrt(slice_var + complement_var)
    return (2**0.5) * (error / std)


def coverage(sample_slice_indices: SliceIdx, slice_base_indices: SliceIdx) -> float:
    mask = np.isin(sample_slice_indices, slice_base_indices)
    true_indices = sample_slice_indices[mask]
    return len(true_indices) / len(sample_slice_indices)
