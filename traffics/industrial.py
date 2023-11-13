import numpy as np

from sixg_radio_mgmt import Traffic


class IndustrialTraffic(Traffic):
    def __init__(
        self,
        max_number_ues: int,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(max_number_ues, rng)

    def step(
        self,
        slice_ue_assoc: np.ndarray,
        slice_req: dict,
        step_number: int,
        episode_number: int,
    ) -> np.ndarray:
        traffic_per_ue = np.zeros(self.max_number_ues)
        num_ue_embb = slice_req["embb"]["number_ues"]
        num_ue_urllc = slice_req["urllc"]["number_ues"]
        num_ue_mmtc = slice_req["mmtc"]["number_ues"]

        # eMBB
        traffic_per_ue[0:num_ue_embb] = (
            self.rng.poisson(slice_req["embb"]["ue_throughput"], num_ue_embb)
            * 1e6
        )  # Mbps
        # URLLC
        traffic_per_ue[num_ue_embb : num_ue_embb + num_ue_urllc] = (
            self.rng.poisson(
                slice_req["urllc"]["ue_throughput"],
                num_ue_urllc,
            )
            * 1e6
        )  # Mbps
        # mMTC
        active_ues = (
            self.rng.random(slice_req["mmtc"]["number_ues"])
            > slice_req["mmtc"]["active_probability"]
        ).astype(int)
        traffic_per_ue[
            num_ue_embb
            + num_ue_urllc : num_ue_embb
            + num_ue_urllc
            + num_ue_mmtc
        ] = active_ues * (
            self.rng.poisson(
                slice_req["mmtc"]["ue_throughput"],
                num_ue_mmtc,
            )
            * 1e6
        )  # Mbps

        return traffic_per_ue
