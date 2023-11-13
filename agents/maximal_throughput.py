# from typing import Union

# import numpy as np

# from sixg_radio_mgmt import Agent


# class MaximalThroughput(Agent):
#     def __init__(
#         self,
#         max_number_ues: int,
#         max_number_basestations: int,
#         num_available_rbs: np.ndarray,
#     ) -> None:
#         super().__init__(max_number_ues, max_number_basestations, num_available_rbs)

#     def step(self, obs_space: Union[np.ndarray, dict]) -> np.ndarray:
#         mt_idx = np.min[obs_space["spectral_efficiencies"]], obs_space["buffer_occupancies"]
#         print(mt_idx.sort(reverse = True))
#         allocation_rbs = [
#             np.zeros((self.max_number_ues, self.num_available_rbs[basestation]))
#           for basestation in np.arange(self.max_number_basestations)
#         ]
#         for basestation in np.arange(self.max_number_basestations):
#             ue_idx = 0
#             rb_idx = 0
#             while rb_idx < self.num_available_rbs[basestation]:
#                 if obs_space[basestation][ue_idx] == 1:
#                     allocation_rbs[basestation][ue_idx][rb_idx] += 1
#                     rb_idx += 1
#                 ue_idx += 1 if ue_idx + 1 != self.max_number_ues else -ue_idx

#         return np.array(allocation_rbs)

#     @staticmethod
#     def obs_space_format(obs_space: dict) -> np.ndarray:
#         return np.array(obs_space["basestation_ue_assoc"])

#     @staticmethod
#     def calculate_reward(obs_space: dict) -> float:
#         return 0


# def main():
#     rr = MaximalThroughput(2, 2, np.array([3, 2]))
#     basestation_ue_assoc = np.array([[1, 1], [1, 1]])
#     for i in range(1):
#         print(rr.step(basestation_ue_assoc))


# if __name__ == "__main__":
#     main()
