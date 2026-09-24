from collections import deque
from typing import Union

import numpy as np

from marl_custom_env import MARLCustomEnv
from sixg_radio_mgmt import MARLCommEnv


def scores_to_rbs(
    action: np.ndarray, total_rbs: int, association: np.ndarray
) -> np.ndarray:
    rbs_per_unit = (
        round_int_equal_sum(
            total_rbs * (action + 1) / np.sum(action + 1),
            total_rbs,
        )
        if np.sum(action + 1) != 0
        else round_int_equal_sum(
            (total_rbs / np.sum(association)) * association,
            total_rbs,
        )
    )
    assert np.sum(rbs_per_unit < 0) == 0, "Negative RBs"
    assert (
        np.sum(rbs_per_unit * association).astype(int) == total_rbs
    ), f"Allocated RBs {np.sum(rbs_per_unit * association)} are different from available RBs {total_rbs}\n{action}\n{rbs_per_unit}\n{association}"

    return rbs_per_unit


def distribute_rbs_ues(
    rbs_per_ue: np.ndarray,
    allocation_rbs: np.ndarray,
    slice_ues: np.ndarray,
    rbs_per_slice: np.ndarray,
    slice_idx: int,
) -> np.ndarray:
    rb_idx = np.sum(rbs_per_slice[:slice_idx], dtype=int)
    for idx, ue_idx in enumerate(slice_ues):
        allocation_rbs[
            0, ue_idx, rb_idx : rb_idx + rbs_per_ue[idx].astype(int)
        ] = 1
        rb_idx += rbs_per_ue[idx].astype(int)

    return allocation_rbs


def round_int_equal_sum(
    float_array: np.ndarray, target_sum: int
) -> np.ndarray:
    non_zero_indices = np.where(float_array != 0)[0]
    non_zero_values = float_array[non_zero_indices]

    # Proportional distribution to get as close as possible to the target sum
    proportional_integers = np.floor(
        target_sum * non_zero_values / np.sum(non_zero_values)
    ).astype(int)

    # Calculate the remaining adjustment
    adjustment = target_sum - np.sum(proportional_integers)

    # Distribute the remaining adjustment among the highest values
    sorted_indices = np.argsort(non_zero_values)[::-1]
    for i in range(adjustment):
        index = sorted_indices[i % len(sorted_indices)]
        proportional_integers[index] += 1

    # Reconstruct the rounded result
    rounded_integers = np.zeros_like(float_array, dtype=int)
    rounded_integers[non_zero_indices] = proportional_integers

    return rounded_integers


def round_robin(
    allocation_rbs: np.ndarray,
    slice_idx: int,
    rbs_per_slice: np.ndarray,
    slice_ues: np.ndarray,
    last_unformatted_obs: deque,
    distribute_rbs: bool = True,
    account_buffer: bool = True,
) -> np.ndarray:
    buffer_occ = last_unformatted_obs[0]["buffer_occupancies"][slice_ues]
    slice_ues_buffer = slice_ues
    if account_buffer:
        slice_ues_buffer = slice_ues[
            np.logical_not(np.isclose(buffer_occ, np.zeros_like(buffer_occ)))
        ]  # Consider only UEs that have packets available in the buffer
        if slice_ues_buffer.shape[0] == 0:
            slice_ues_buffer = slice_ues
    rbs_per_ue = np.ones_like(slice_ues_buffer, dtype=float) * np.floor(
        rbs_per_slice[slice_idx] / slice_ues_buffer.shape[0]
    )
    remaining_rbs = int(rbs_per_slice[slice_idx] % slice_ues_buffer.shape[0])
    rbs_per_ue[0:remaining_rbs] += 1
    assert (
        np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
    ), "RR: Number of allocated RBs is different than available RBs"
    assert (
        np.sum(rbs_per_ue < 0) == 0
    ), "Negative RBs on rbs_per_ue are not allowed"

    if distribute_rbs:
        allocation_rbs = distribute_rbs_ues(
            rbs_per_ue,
            allocation_rbs,
            slice_ues_buffer,
            rbs_per_slice,
            slice_idx,
        )
        assert (
            np.sum(allocation_rbs[0, slice_ues_buffer, :])
            == rbs_per_slice[slice_idx]
        ), "Distribute RBs is different from RR distribution"
        assert np.sum(allocation_rbs) == np.sum(
            rbs_per_slice[0 : slice_idx + 1]
        ), f"allocation_rbs is different from rbs_per_slice at slice {slice_idx}"

        return allocation_rbs
    else:
        return rbs_per_ue


def proportional_fairness(
    allocation_rbs: np.ndarray,
    slice_idx: int,
    rbs_per_slice: np.ndarray,
    slice_ues: np.ndarray,
    env: Union[MARLCustomEnv, MARLCommEnv],
    last_unformatted_obs: deque,
    num_available_rbs: np.ndarray,
) -> np.ndarray:
    spectral_eff = np.mean(
        last_unformatted_obs[0]["spectral_efficiencies"][0, slice_ues, :],
        axis=1,
    )
    buffer_occ = last_unformatted_obs[0]["buffer_occupancies"][slice_ues]
    throughput_available = np.minimum(
        spectral_eff
        * (
            float(rbs_per_slice[slice_idx])
            * float(env.comm_env.bandwidths[0])
            / float(num_available_rbs[0])
        )
        / slice_ues.shape[0],
        buffer_occ
        * env.comm_env.ues.max_buffer_pkts[slice_ues]
        * env.comm_env.ues.pkt_sizes[slice_ues],
    )
    pkt_snt_throughput = np.mean(
        [
            last_unformatted_obs[idx]["pkt_effective_thr"][slice_ues]
            for idx in range(len(last_unformatted_obs))
        ],
        axis=0,
    )
    snt_throughput = pkt_snt_throughput * env.comm_env.ues.pkt_sizes[slice_ues]
    snt_throughput[
        np.isclose(throughput_available, np.zeros_like(throughput_available))
    ] = 1
    weights = np.divide(
        throughput_available,
        snt_throughput,
        where=np.logical_not(
            np.isclose(snt_throughput, np.zeros_like(snt_throughput))
        ),
        out=2 * np.max(throughput_available) * np.ones_like(snt_throughput),
    )
    rbs_per_ue = (
        round_int_equal_sum(
            rbs_per_slice[slice_idx] * weights / np.sum(weights),
            int(rbs_per_slice[slice_idx]),
        )
        if np.sum(weights) != 0
        else round_robin(
            allocation_rbs=np.array([]),
            slice_idx=slice_idx,
            rbs_per_slice=rbs_per_slice,
            slice_ues=slice_ues,
            last_unformatted_obs=last_unformatted_obs,
            distribute_rbs=False,
            account_buffer=False,
        )
    )
    allocation_rbs = distribute_rbs_ues(
        rbs_per_ue, allocation_rbs, slice_ues, rbs_per_slice, slice_idx
    )

    assert (
        np.sum(rbs_per_ue < 0) == 0
    ), "Negative RBs on rbs_per_ue are not allowed"
    assert (
        np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
    ), "PF: Number of allocated RBs is different than available RBs"
    assert (
        np.sum(allocation_rbs[0, slice_ues, :]) == rbs_per_slice[slice_idx]
    ), "Distribute RBs is different from RR distribution"
    assert np.sum(allocation_rbs) == np.sum(
        rbs_per_slice[0 : slice_idx + 1]
    ), f"allocation_rbs is different from rbs_per_slice at slice {slice_idx}"

    return allocation_rbs


def max_throughput(
    allocation_rbs: np.ndarray,
    slice_idx: int,
    rbs_per_slice: np.ndarray,
    slice_ues: np.ndarray,
    env: Union[MARLCustomEnv, MARLCommEnv],
    last_unformatted_obs: deque,
    num_available_rbs: np.ndarray,
) -> np.ndarray:
    spectral_eff = np.mean(
        last_unformatted_obs[0]["spectral_efficiencies"][0, slice_ues, :],
        axis=1,
    )
    buffer_occ = last_unformatted_obs[0]["buffer_occupancies"][slice_ues]
    throughput_available = np.minimum(
        spectral_eff
        * (
            float(rbs_per_slice[slice_idx])
            * float(env.comm_env.bandwidths[0])
            / float(num_available_rbs[0])
        )
        / slice_ues.shape[0],
        buffer_occ
        * env.comm_env.ues.max_buffer_pkts[slice_ues]
        * env.comm_env.ues.pkt_sizes[slice_ues],
    )
    rbs_per_ue = (
        round_int_equal_sum(
            rbs_per_slice[slice_idx]
            * throughput_available
            / np.sum(throughput_available),
            int(rbs_per_slice[slice_idx]),
        )
        if np.sum(throughput_available) != 0
        else round_robin(
            allocation_rbs=np.array([]),
            slice_idx=slice_idx,
            rbs_per_slice=rbs_per_slice,
            slice_ues=slice_ues,
            last_unformatted_obs=last_unformatted_obs,
            distribute_rbs=False,
            account_buffer=False,
        )
    )
    allocation_rbs = distribute_rbs_ues(
        rbs_per_ue, allocation_rbs, slice_ues, rbs_per_slice, slice_idx
    )

    assert (
        np.sum(rbs_per_ue < 0) == 0
    ), "Negative RBs on rbs_per_ue are not allowed"
    assert (
        np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
    ), "MT: Number of allocated RBs is different than available RBs"
    assert (
        np.sum(allocation_rbs[0, slice_ues, :]) == rbs_per_slice[slice_idx]
    ), "Distribute RBs is different from RR distribution"

    assert np.sum(allocation_rbs) == np.sum(
        rbs_per_slice[0 : slice_idx + 1]
    ), f"allocation_rbs is different from rbs_per_slice at slice {slice_idx}"

    return allocation_rbs
