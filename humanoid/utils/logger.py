# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process
import json
import pickle
from pathlib import Path


class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if "rew" in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot joint targets and measured positions
        a = axs[1, 0]
        if log["dof_pos"]:
            a.plot(time, log["dof_pos"], label="measured")
        if log["dof_pos_target"]:
            a.plot(time, log["dof_pos_target"], label="target")
        a.set(xlabel="time [s]", ylabel="Position [rad]", title="DOF Position")
        a.legend()
        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel"]:
            a.plot(time, log["dof_vel"], label="measured")
        if log["dof_vel_target"]:
            a.plot(time, log["dof_vel_target"], label="target")
        a.set(xlabel="time [s]", ylabel="Velocity [rad/s]", title="Joint Velocity")
        a.legend()
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]:
            a.plot(time, log["base_vel_x"], label="measured")
        if log["command_x"]:
            a.plot(time, log["command_x"], label="commanded")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity x")
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]:
            a.plot(time, log["base_vel_y"], label="measured")
        if log["command_y"]:
            a.plot(time, log["command_y"], label="commanded")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity y")
        a.legend()
        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]:
            a.plot(time, log["base_vel_yaw"], label="measured")
        if log["command_yaw"]:
            a.plot(time, log["command_yaw"], label="commanded")
        a.set(
            xlabel="time [s]", ylabel="base ang vel [rad/s]", title="Base velocity yaw"
        )
        a.legend()
        # plot base vel z
        a = axs[1, 2]
        if log["base_vel_z"]:
            a.plot(time, log["base_vel_z"], label="measured")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity z")
        a.legend()
        # plot contact forces
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f"force {i}")
        a.set(xlabel="time [s]", ylabel="Forces z [N]", title="Vertical Contact forces")
        a.legend()
        # plot torque/vel curves
        a = axs[2, 1]
        if log["dof_vel"] != [] and log["dof_torque"] != []:
            a.plot(log["dof_vel"], log["dof_torque"], "x", label="measured")
        a.set(
            xlabel="Joint vel [rad/s]",
            ylabel="Joint Torque [Nm]",
            title="Torque/velocity curves",
        )
        a.legend()
        # plot torques
        a = axs[2, 2]
        if log["dof_torque"] != []:
            a.plot(time, log["dof_torque"], label="measured")
        a.set(xlabel="time [s]", ylabel="Joint Torque [Nm]", title="Torque")
        a.legend()
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()

    def save(self, filename: str, format: str = "json", save_dir: str = "./logs"):
        """
        保存日志数据到文件
        Args:
            filename: 文件名（无需后缀）
            format: 保存格式，支持 json/pkl
            save_dir: 存储目录
        """
        # 确保目录存在
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        path = Path(save_dir) / f"{filename}.{format}"

        # 转换为可序列化的字典
        data = {
            "dt": self.dt,
            "state_log": self._convert_log(self.state_log),
            "rew_log": self._convert_log(self.rew_log),
            "num_episodes": self.num_episodes,
        }

        # 选择保存格式
        if format == "json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif format == "pkl":
            with open(path, "wb") as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load(self, filepath: str):
        """
        从文件加载日志数据
        Args:
            filepath: 文件路径（需要包含后缀）
        """
        format = Path(filepath).suffix[1:]

        # 读取数据
        if format == "json":
            with open(filepath, "r") as f:
                data = json.load(f)
        elif format == "pkl":
            with open(filepath, "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # 恢复状态
        self.dt = data["dt"]
        self.state_log = defaultdict(list, data["state_log"])
        self.rew_log = defaultdict(list, data["rew_log"])
        self.num_episodes = data["num_episodes"]

    def _convert_log(self, log):
        """将日志中的张量/数组转为Python原生类型"""
        converted = {}
        for key, values in log.items():
            # 处理PyTorch张量 -> numpy数组 -> Python列表
            if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                if hasattr(values[0], "cpu"):  # 处理PyTorch张量
                    converted[key] = [v.cpu().numpy().tolist() for v in values]
                elif isinstance(values[0], np.ndarray):  # 处理numpy数组
                    converted[key] = [v.tolist() for v in values]
                else:
                    converted[key] = values
            else:
                converted[key] = values
        return converted
