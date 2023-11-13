import os
from typing import Tuple
from matplotlib import lines

import matplotlib.figure as matfig
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import FixedLocator, NullFormatter


def gen_results(
    scenario_names: list[str],
    agent_names: list[str],
    episodes: np.ndarray,
    metrics: list,
    slices: np.ndarray,
):
    xlabel = ylabel = ""
    for scenario in scenario_names:
        for episode in episodes:
            for metric in metrics:
                plt.figure()
                w, h = matfig.figaspect(0.6)
                plt.figure(figsize=(w, h))
                for agent in agent_names:
                    (xlabel, ylabel) = plot_graph(
                        metric, slices, agent, scenario, episode
                    )
                plt.grid()
                plt.xlabel(xlabel, fontsize=14)
                plt.ylabel(ylabel, fontsize=14)
                plt.xticks(fontsize=12)
                plt.legend(fontsize=12)
                os.makedirs(
                    f"./results/{scenario}/ep_{episode}",
                    exist_ok=True,
                )
                plt.savefig(
                    "./results/{}/ep_{}/{}.pdf".format(
                        scenario, episode, metric
                    ),
                    bbox_inches="tight",
                    pad_inches=0,
                    format="pdf",
                    dpi=1000,
                )
                plt.close("all")


def plot_graph(
    metric: str,
    slices: np.ndarray,
    agent: str,
    scenario: str,
    episode: int,
) -> Tuple[str, str]:
    xlabel = ylabel = ""
    data = np.load(
        f"hist/{scenario}/{agent}/ep_{episode}.npz",
        allow_pickle=True,
    )
    data_metrics = {
        "pkt_incoming": data["pkt_incoming"],
        "pkt_throughputs": data["pkt_throughputs"],
        "pkt_effective_thr": data["pkt_effective_thr"],
        "buffer_occupancies": data["buffer_occupancies"],
        "buffer_latencies": data["buffer_latencies"],
        "dropped_pkts": data["dropped_pkts"],
        "mobility": data["mobility"],
        "spectral_efficiencies": data["spectral_efficiencies"],
        "basestation_ue_assoc": data["basestation_ue_assoc"],
        "basestation_slice_assoc": data["basestation_slice_assoc"],
        "slice_ue_assoc": data["slice_ue_assoc"],
        "sched_decision": data["sched_decision"],
        "reward": data["reward"],
        "slice_req": data["slice_req"],
    }
    for slice in slices:
        match metric:
            case (
                "pkt_incoming"
                | "pkt_effective_thr"
                | "pkt_throughputs"
                | "dropped_pkts"
            ):
                slice_throughput = calc_throughput_slice(
                    data_metrics, metric, slice
                )
                plt.plot(slice_throughput, label=f"{agent}, slice {slice}")
                xlabel = "Step (n)"
                ylabel = "Throughput (Mbps)"
            case ("buffer_latencies" | "buffer_occupancies"):
                avg_spectral_efficiency = calc_slice_average(
                    data_metrics, metric, slice
                )
                plt.plot(
                    avg_spectral_efficiency,
                    label=f"{agent}, slice {slice}",
                )
                xlabel = "Step (n)"
                match metric:
                    case "buffer_latencies":
                        ylabel = "Average buffer delay (ms)"
                    case "buffer_occupancies":
                        ylabel = "Buffer occupancy rate"
            case ("basestation_ue_assoc" | "basestation_slice_assoc"):
                number_elements = np.sum(
                    np.sum(data_metrics[metric], axis=2), axis=1
                )
                plt.plot(number_elements, label=f"{agent}")
                xlabel = "Step (n)"
                match metric:
                    case "basestation_ue_assoc":
                        ylabel = "Number of UEs"
                    case "basestation_slice_assoc":
                        ylabel = "Number of slices"
                break
            case "slice_allocation":
                slice_allocation = np.sum(
                    np.sum(np.squeeze(data_metrics["sched_decision"]), axis=2)
                    * data_metrics["slice_ue_assoc"][:, slice, :],
                    axis=1,
                )
                plt.plot(slice_allocation, label=f"{agent}, slice {slice}")
                xlabel = "Step (n)"
                ylabel = "# Allocated RBs"
            case "slice_ue_assoc":
                number_uers_per_slice = np.sum(
                    data_metrics[metric][:, slice, :], axis=1
                )
                plt.plot(
                    number_uers_per_slice, label=f"{agent}, slice {slice}"
                )
                xlabel = "Step (n)"
                ylabel = "Number of UEs"
            case "reward":
                plt.plot(data_metrics[metric], label=f"{agent}")
                xlabel = "Step (n)"
                ylabel = "Reward"
                break
            case "reward_cumsum":
                plt.plot(np.cumsum(data_metrics["reward"]), label=f"{agent}")
                xlabel = "Step (n)"
                ylabel = "Cumulative reward"
                break
            case "total_network_throughput":
                total_throughput = calc_total_throughput(
                    data_metrics, "pkt_throughputs"
                )
                plt.plot(total_throughput, label=f"{agent}")
                xlabel = "Step (n)"
                ylabel = "Throughput (Mbps)"
                break
            case "total_network_requested_throughput":
                total_throughput = calc_total_throughput(
                    data_metrics, "pkt_incoming"
                )
                plt.plot(total_throughput, label=f"{agent}")
                xlabel = "Step (n)"
                ylabel = "Throughput (Mbps)"
                break
            case "spectral_efficiencies":
                if slice not in [11]:
                    slice_ues = data_metrics["slice_ue_assoc"][:, slice, :]
                    num = (
                        np.sum(
                            np.sum(np.squeeze(data_metrics[metric]), axis=2)
                            * slice_ues,
                            axis=1,
                        )
                        * 100
                        / 135
                    )
                    den = np.sum(slice_ues, axis=1)
                    spectral_eff = np.divide(
                        num,
                        den,
                        where=np.logical_not(
                            np.isclose(den, np.zeros_like(den))
                        ),
                    )
                    plt.plot(spectral_eff, label=f"{agent}, slice {slice}")
                    xlabel = "Step (n)"
                    ylabel = "Thoughput capacity per RB (Mbps)"
            case "violations":
                violations = calc_slice_violations(data_metrics)
                plt.plot(
                    np.sum(violations[:1, :], axis=0),
                    label=f"{agent}, slice 0",
                )
                plt.plot(
                    np.sum(violations[2:4, :], axis=0),
                    label=f"{agent}, slice 1",
                )
                plt.plot(violations[4, :], label=f"{agent}, slice 2")
                plt.plot(np.sum(violations, axis=0), label=f"{agent}, total")
                xlabel = "Step (n)"
                ylabel = "# Violations"
                break
            case "violations_cumsum":
                violations = calc_slice_violations(data_metrics)
                range_violations = np.arange(1, violations.shape[1] + 1)
                plt.plot(
                    np.cumsum(
                        np.sum(
                            violations[:2, :],
                            axis=0,
                        )
                    )
                    / (2 * range_violations),
                    label=f"{agent}, slice 0",
                )
                plt.plot(
                    np.cumsum(
                        np.sum(
                            violations[2:4, :],
                            axis=0,
                        )
                    )
                    / (2 * range_violations),
                    label=f"{agent}, slice 1",
                )
                plt.plot(
                    np.cumsum(violations[4, :]) / range_violations,
                    label=f"{agent}, slice 2",
                )
                plt.plot(
                    np.cumsum(np.sum(violations, axis=0))
                    / (5 * range_violations),
                    label=f"{agent}, total",
                )
                xlabel = "Step (n)"
                ylabel = "Cumulative # violations"
                break
            case _:
                raise Exception("Metric not found")

    return (xlabel, ylabel)


def calc_throughput_slice(
    data_metrics: dict, metric: str, slice: int
) -> np.ndarray:
    message_sizes = 8192 * 8
    den = np.sum(data_metrics["slice_ue_assoc"][:, slice, :], axis=1)
    slice_throughput = np.divide(
        np.sum(
            (
                data_metrics[metric]
                * data_metrics["slice_ue_assoc"][:, slice, :]
            ),
            axis=1,
        )
        * message_sizes,
        (1e6 * den),
        where=np.logical_not(np.isclose(den, np.zeros_like(den))),
    )

    return slice_throughput


def calc_total_throughput(data_metrics: dict, metric: str) -> np.ndarray:
    message_sizes = 8192 * 8
    slice_throughput = (
        np.sum(
            (data_metrics[metric]),
            axis=1,
        )
        * message_sizes
        / 1e6
    )

    return slice_throughput


def calc_slice_average(
    data_metrics: dict, metric: str, slice: int
) -> np.ndarray:
    slice_ues = data_metrics["slice_ue_assoc"][:, slice, :]
    num_slice_ues = np.sum(slice_ues, axis=1)
    result_values = np.divide(
        np.sum(data_metrics[metric] * slice_ues, axis=1),
        num_slice_ues,
        where=np.logical_not(
            np.isclose(num_slice_ues, np.zeros_like(num_slice_ues))
        ),
    )

    return result_values


def calc_slice_violations(data_metrics) -> np.ndarray:
    # 1st dimension: eMBB throughput violations
    # 2nd dimension: eMBB latency violations
    # 3rd dimension: URLLC throughput violations
    # 4th dimension: URLLC latency violations
    # 5th dimension: mMTC latency violations
    violations = np.zeros((5, data_metrics["pkt_throughputs"].shape[0]))

    # Requirements
    with open("./env_config/industrial.yml") as file:
        data = yaml.safe_load(file)
    embb_throughput_req = data["slices"]["slice_req"]["embb"]["ue_throughput"]
    embb_latency_req = data["slices"]["slice_req"]["embb"]["latency"]
    urllc_throughput_req = data["slices"]["slice_req"]["urllc"][
        "ue_throughput"
    ]
    urllc_latency_req = data["slices"]["slice_req"]["urllc"]["latency"]
    mmtc_latency_req = data["slices"]["slice_req"]["mmtc"]["latency"]

    # eMBB violations
    violations[0, :] = (
        calc_throughput_slice(data_metrics, "pkt_throughputs", 0)
        < embb_throughput_req
    ).astype(int)
    violations[1, :] = (
        calc_slice_average(data_metrics, "buffer_latencies", 0)
        > embb_latency_req
    ).astype(int)

    # URLLC violations
    violations[2, :] = (
        calc_throughput_slice(data_metrics, "pkt_throughputs", 1)
        < urllc_throughput_req
    ).astype(int)
    violations[3, :] = (
        calc_slice_average(data_metrics, "buffer_latencies", 1)
        > urllc_latency_req
    ).astype(int)

    # mMTC violations
    violations[4, :] = (
        calc_slice_average(data_metrics, "buffer_latencies", 2)
        > mmtc_latency_req
    ).astype(int)

    return violations


def gen_results_violations(
    scenario_names: list[str],
    agent_names: list[str],
    episodes: np.ndarray,
    slice_names: list[str],
):
    xlabel = "TTI"
    ylabel = "# of slice violations"
    for scenario in scenario_names:
        plt.figure()
        w, h = matfig.figaspect(0.6)
        plt.figure(figsize=(w, h))
        for agent in agent_names:
            episodes_violations = read_episodes_violations(
                scenario, agent, episodes
            )
            slice_metrics_selection = {
                "embb": np.array([0, 1]),
                "urllc": np.array([2, 3]),
                "mmtc": np.array([4]),
                "total": np.array([0, 1, 2, 3, 4]),
            }
            for slice in slice_names:
                slice_episodes_violations = np.sum(
                    episodes_violations[:, slice_metrics_selection[slice], :],
                    axis=1,
                )
                mean_violations = np.mean(slice_episodes_violations, axis=0)
                std_violations = np.std(slice_episodes_violations, axis=0)
                agent_name = "SSR" if agent == "ssr" else "Proposed Method"
                plt.plot(mean_violations, label=f"{agent_name}, {slice}")
                plt.fill_between(
                    np.arange(std_violations.shape[0]),
                    mean_violations - std_violations,
                    mean_violations + std_violations,
                    alpha=0.2,
                )
        plt.grid()
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        os.makedirs(
            f"./results/{scenario}/",
            exist_ok=True,
        )
        plt.savefig(
            "./results/{}/violations_analisys.pdf".format(scenario),
            bbox_inches="tight",
            pad_inches=0,
            format="pdf",
            dpi=1000,
        )
        plt.close("all")


def read_episodes_violations(
    scenario: str,
    agent: str,
    episodes: np.ndarray,
) -> np.ndarray:
    num_slice_requirements = 5
    steps_per_episode = 1000
    episodes_violations = np.zeros(
        (
            episodes.shape[0],
            num_slice_requirements,
            steps_per_episode,
        )
    )
    for idx, episode in enumerate(episodes):
        data = np.load(
            f"hist/{scenario}/{agent}/ep_{episode}.npz",
            allow_pickle=True,
        )
        data_metrics = {
            "pkt_incoming": data["pkt_incoming"],
            "pkt_throughputs": data["pkt_throughputs"],
            "pkt_effective_thr": data["pkt_effective_thr"],
            "buffer_occupancies": data["buffer_occupancies"],
            "buffer_latencies": data["buffer_latencies"],
            "dropped_pkts": data["dropped_pkts"],
            "mobility": data["mobility"],
            "spectral_efficiencies": data["spectral_efficiencies"],
            "basestation_ue_assoc": data["basestation_ue_assoc"],
            "basestation_slice_assoc": data["basestation_slice_assoc"],
            "slice_ue_assoc": data["slice_ue_assoc"],
            "sched_decision": data["sched_decision"],
            "reward": data["reward"],
            "slice_req": data["slice_req"],
        }
        episodes_violations[idx, :, :] = calc_slice_violations(data_metrics)

    return episodes_violations


def gen_results_histogram(
    scenario_names: list[str],
    agent_names: list[str],
    episodes: np.ndarray,
    slice_names: list[str],
    metrics: list[str],
):
    slice_idx = {
        "embb": 0,
        "urllc": 1,
        "mmtc": 2,
    }
    color = {"ssr_protect": "tab:blue", "ssr": "tab:red"}
    markers = {"embb": "o", "urllc": "s", "mmtc": "*"}
    ylabel = "Cumulative distribution function (CDF)"
    xlabel = ""
    for scenario in scenario_names:
        for metric in metrics:
            if metric in [
                "pkt_incoming",
                "pkt_throughputs",
                "pkt_effective_thr",
            ]:
                xlabel = "Throughput (Mbps)"
            elif metric in ["buffer_latencies"]:
                xlabel = "Buffer delay (ms)"
            plt.figure()
            w, h = matfig.figaspect(0.6)
            plt.figure(figsize=(w, h))
            for agent in agent_names:
                (episode_metrics, slice_ue_assoc) = read_episode_metric(
                    scenario, agent, episodes, metric
                )
                # episode_metrics = [episodes, steps_per_episode, number_ues]
                # slice_ue_assoc = [episodes, steps_per_episode, number_slices, number_ues]
                for slice in slice_names:
                    message_sizes = (
                        8192 * 8
                        if metric
                        in [
                            "pkt_incoming",
                            "pkt_throughputs",
                            "pkt_effective_thr",
                        ]
                        else 1
                    )
                    den_throughput = (
                        1e6
                        if metric
                        in [
                            "pkt_incoming",
                            "pkt_throughputs",
                            "pkt_effective_thr",
                        ]
                        else 1
                    )
                    slice_values = (
                        np.sum(
                            episode_metrics
                            * slice_ue_assoc[:, :, slice_idx[slice], :],
                            axis=2,
                        )
                        * message_sizes
                    ) / (
                        den_throughput
                        * np.sum(
                            slice_ue_assoc[:, :, slice_idx[slice], :],
                            axis=2,
                        )
                    )
                    slice_values = np.sort(
                        slice_values[slice_values != 0].flatten()
                    )
                    y_axis = (
                        np.arange(slice_values.shape[0])
                        / slice_values.shape[0]
                    )
                    agent_name = "SSR" if agent == "ssr" else "Proposed Method"
                    plt.plot(
                        slice_values,
                        y_axis,
                        label=f"{agent_name}, {slice}",
                        color=color[agent],
                        marker=markers[slice],
                        linestyle="-",
                        markevery=4000,
                    )
            if metric in ["buffer_latencies", "pkt_throughputs"]:
                small_scale_interval = 1
                large_scale_interval = 5
                small_scale_limit = 5
                graph_x_limit = 50 if metric == "buffer_latencies" else 27
                try:
                    assert isinstance(
                        slice_values, np.ndarray  # type: ignore
                    ), "Slice values must be a Numpy array"
                    locator, tick_labels = custom_grid_locator(
                        np.arange(np.max(slice_values)),
                        small_scale_interval,
                        large_scale_interval,
                        small_scale_limit,
                        graph_x_limit,
                    )
                    plt.gca().xaxis.set_major_locator(locator)
                    plt.gca().xaxis.set_ticklabels(tick_labels)
                except ValueError:
                    slice_values = None
            plt.grid()
            plt.xlabel(xlabel, fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.xticks(fontsize=12)
            if metric in ["buffer_latencies"]:
                plt.legend(fontsize=12, loc="lower right")
            else:
                plt.legend(fontsize=12)
            os.makedirs(
                f"./results/{scenario}/",
                exist_ok=True,
            )
            plt.savefig(
                f"./results/{scenario}/{metric}.pdf",
                bbox_inches="tight",
                pad_inches=0,
                format="pdf",
                dpi=1000,
            )
            plt.close("all")


def read_episode_metric(
    scenario: str,
    agent: str,
    episodes: np.ndarray,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    steps_per_episode = 1000
    number_ues = 100
    number_slices = 3
    episode_metric = np.zeros(
        (
            episodes.shape[0],
            steps_per_episode,
            number_ues,
        )
    )
    slice_ue_assoc = np.zeros(
        (
            episodes.shape[0],
            steps_per_episode,
            number_slices,
            number_ues,
        )
    )
    for idx, episode in enumerate(episodes):
        data = np.load(
            f"hist/{scenario}/{agent}/ep_{episode}.npz",
            allow_pickle=True,
        )
        episode_metric[idx, :, :] = data[metric]
        slice_ue_assoc[idx, :, :, :] = data["slice_ue_assoc"]

    return (episode_metric, slice_ue_assoc)


def custom_grid_locator(
    axis,
    small_scale_interval,
    large_scale_interval,
    small_scale_limit,
    graph_x_limit,
):
    small_scale_ticks = np.arange(0, small_scale_limit, small_scale_interval)
    large_scale_ticks = np.arange(
        small_scale_limit, graph_x_limit, large_scale_interval
    )

    all_ticks = np.concatenate((small_scale_ticks, large_scale_ticks))

    tick_labels = [
        str(tick)
        if tick in large_scale_ticks or tick == 0 or tick == 1
        else ""
        for tick in all_ticks
    ]

    return FixedLocator(all_ticks), tick_labels


scenario_names = ["industrial"]
agent_names = ["ssr_protect", "ssr"]
metrics = [
    "pkt_incoming",
    "pkt_effective_thr",
    "pkt_throughputs",
    "dropped_pkts",
    "buffer_occupancies",
    "buffer_latencies",
    "basestation_ue_assoc",
    "basestation_slice_assoc",
    "slice_ue_assoc",
    "reward",
    "reward_cumsum",
    "total_network_throughput",
    "spectral_efficiencies",
    "violations",
    "violations_cumsum",
    "total_network_requested_throughput",
    "slice_allocation",
]
episodes = np.arange(190, 200)
slices = np.arange(3)

# gen_results(scenario_names, agent_names, episodes, metrics, slices)
episodes = np.arange(160, 200)
slice_names = ["urllc", "total"]
agent_names = ["ssr_protect", "ssr"]
gen_results_violations(scenario_names, agent_names, episodes, slice_names)
metrics = ["buffer_latencies", "pkt_throughputs"]
slice_names = ["embb", "urllc", "mmtc"]
gen_results_histogram(
    scenario_names,
    agent_names,
    episodes,
    slice_names,
    metrics,
)
