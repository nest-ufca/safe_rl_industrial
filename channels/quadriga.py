import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio

from sixg_radio_mgmt import Channel


class QuadrigaChannels(Channel):
    def __init__(
        self,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        rng: np.random.Generator = np.random.default_rng(),
        root_path: str = "",
        scenario_name: str = "",
    ) -> None:
        super().__init__(
            max_number_ues, max_number_basestations, num_available_rbs, rng
        )
        self.thermal_noise_power = 10e-14
        self.transmission_power = 0.1  # 0.1 Watts = 20 dBm
        self.episode_number = -1
        self.episode_channels = np.empty([])

    def step(
        self,
        step_number: int,
        episode_number: int,
        mobilities: np.ndarray,
        sched_decision: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.episode_number != episode_number:
            self.episode_number = episode_number
            self.episode_channels = self.read_mat_files(episode_number)

        if sched_decision is not None:
            return self.calc_spectral_eff(sched_decision, step_number)
        else:
            spectral_efficiencies = [
                np.ones((self.max_number_ues, self.num_available_rbs[i]))
                for i in np.arange(self.max_number_basestations)
            ]
        return np.array(spectral_efficiencies)

    def calc_spectral_eff(
        self, sched_decision: np.ndarray, step_number: int
    ) -> np.ndarray:
        # Implementing the MRT precoder to achieve diversity gains
        x = self.episode_channels.shape
        allocated_rbs_channels = np.zeros(
            (1, self.max_number_ues, x[3]), dtype="complex"
        )
        for u in range(x[2]):
            for rb in range(x[3]):
                channel_herm = self.episode_channels[
                    :, :, u, rb, step_number
                ].conj()
                channel_norm = np.linalg.norm(channel_herm)
                w = channel_herm / channel_norm
                aux = np.dot(
                    self.episode_channels[:, :, u, rb, step_number], w.conj().T
                )
                allocated_rbs_channels[0, u, rb] = aux[0, 0]

        # Sum transmitter and receiver antennas, and changing dimensions to
        # B x U x R to match sched_decision dimensions. After multiplying
        # with sched_decision, we obtain the channels only in the allocated RB
        allocated_rbs_channels = allocated_rbs_channels * sched_decision

        allocated_rbs_rsrp = np.power(np.abs(allocated_rbs_channels), 2)
        spectral_efficiencies = np.log2(
            1
            + (
                (self.transmission_power / self.num_available_rbs[0])
                * allocated_rbs_rsrp
                / self.thermal_noise_power
            )
        )
        return spectral_efficiencies

    def read_mat_files(self, episode: int) -> np.ndarray:
        default_dir = Path(__file__).resolve().parent / "quadriga_channels"
        channel_dir = Path(
            os.environ.get("QUADRIGA_CHANNEL_DIR", str(default_dir))
        )
        channel_file = channel_dir / f"sim_{episode + 1}.mat"
        if not channel_file.is_file():
            raise FileNotFoundError(
                f"Canal QuaDRiGa não encontrado: {channel_file}. "
                "Defina QUADRIGA_CHANNEL_DIR com o diretório dos arquivos .mat."
            )
        channels = sio.loadmat(channel_file)

        return channels["H"]
