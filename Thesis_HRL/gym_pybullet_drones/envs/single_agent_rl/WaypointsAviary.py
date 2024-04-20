import os
import numpy as np
import math
import pybullet as p
import pkg_resources
from typing import List

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
    BaseSingleAgentAviary,
)

from gym_pybullet_drones.envs.single_agent_rl.rewards import getRewardDict
from gym_pybullet_drones.envs.single_agent_rl.terminations import getTermDict
from gym_pybullet_drones.envs.single_agent_rl.rewards import EnterAreaReward, IncreaseXReward, CollisionReward, WaypointReward
from gym_pybullet_drones.envs.single_agent_rl.terminations import BoundsTerm, OrientationTerm, CollisionTerm


class WaypointsAviary(BaseSingleAgentAviary):
    """Single agent RL problem: fly through waypoints."""

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        freq: int = 240,
        aggregate_phy_steps: int = 1,
        gui=False,
        record=False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RPM,
        difficulty = 0,
        bounds: List = [[5, 5, 5], [-5, -5, 0.1]],
        collision_detection = False
    ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.bounds = bounds

        self.obstacles = []
        self.reward_components = []

        if len(self.reward_components) == 0:
            self.reward_components.append(WaypointReward(scale=10, waypoints=[[0,0,1]]))
            # self.reward_components.append(
            #     EnterAreaReward(scale=10, area=[[4, 5], [-1, 1]])
            # )
            # self.reward_components.append(IncreaseXReward(scale=0.1))

        self.term_components = []
        if len(self.term_components) == 0:
            self.term_components.append(BoundsTerm(bounds=self.bounds))
            self.term_components.append(OrientationTerm())

        super().__init__(
            drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

        # override base aviary episode length
        self.EPISODE_LEN_SEC = 20
        self.difficulty = difficulty

        if collision_detection:
            self.term_components.append(CollisionTerm([[11, 12], [-1, 1]], self.CLIENT))
            self.reward_components.append(
                CollisionReward(0.2, [[4, 5], [-1, 1]], self.CLIENT)
            )

        self.reward_dict = getRewardDict(self.reward_components)
        self.term_dict = getTermDict(self.term_components)
        self.cum_reward_dict = getRewardDict(self.reward_components)
        self.truncated = False
        self.done = False
        

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Extends the superclass method and add the gate build of cubes and an architrave.

        """
        super()._addObstacles()
        num_targets = 5
        for obstacle in self.obstacles:
            obstacle._addObstacles()
        # we sample from polar coordinates to generate linear targets
        targets = np.zeros(shape=(num_targets, 3))
        thetas = np.random.uniform(0.0, 2.0 * math.pi, size=(num_targets,))
        phis = np.random.uniform(0.0, 2.0 * math.pi, size=(num_targets,))
        for i, theta, phi in zip(range(num_targets), thetas, phis):
            dist = np.random.uniform(low=1.0, high=5 * 0.9)
            x = dist * math.sin(phi) * math.cos(theta)
            y = dist * math.sin(phi) * math.sin(theta)
            z = abs(dist * math.cos(phi))

            # check for floor of z
            targets[i] = np.array([x, y, z if z > 0.1 else 0.1])


       
        target_visual = []
        for target in targets:
            target_visual.append(
                p.loadURDF(
                    os.path.dirname(os.path.abspath(__file__))
                    + "/../../assets/waypoint.urdf",
                    basePosition=target,
                    useFixedBase=True,
                    globalScaling=0.05,
                )
            )

        for i, visual in enumerate(target_visual):
            p.changeVisualShape(
                visual,
                linkIndex=-1,
                rgbaColor=(0, 1 - (i / len(target_visual)), 0, 1),
            )
        

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        norm_ep_time = (self.step_counter / self.SIM_FREQ) / self.EPISODE_LEN_SEC
        # return (
        #     -10
        #     * np.linalg.norm(np.array([0, -2 * norm_ep_time, 0.75]) - state[0:3]) ** 2
        # )
        return -1 * np.linalg.norm(np.array([0, 0, 1])-state[0:3])**2

    ################################################################################

    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
            self.truncated = True
            self.done = True
        else:
            state = self._getDroneStateVector(0)
            done = False
            for term_component, t_dict in zip(self.term_components, self.term_dict):
                t = term_component.calculateTerm(state)
                self.term_dict[t_dict] = t
                done = done or t
                if done:
                    print(t_dict)
                    print(state)
            self.done = done

        return self.done

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {
            "answer": 42
        }  #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def _clipAndNormalizeState(self, state):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(
                state,
                clipped_pos_xy,
                clipped_pos_z,
                clipped_rp,
                clipped_vel_xy,
                clipped_vel_z,
            )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = (
            state[13:16] / np.linalg.norm(state[13:16])
            if np.linalg.norm(state[13:16]) != 0
            else state[13:16]
        )

        norm_and_clipped = np.hstack(
            [
                normalized_pos_xy,
                normalized_pos_z,
                state[3:7],
                normalized_rp,
                normalized_y,
                normalized_vel_xy,
                normalized_vel_z,
                normalized_ang_vel,
                state[16:20],
            ]
        ).reshape(20,)

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(
        self,
        state,
        clipped_pos_xy,
        clipped_pos_z,
        clipped_rp,
        clipped_vel_xy,
        clipped_vel_z,
    ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(
                    state[0], state[1]
                ),
            )
        if not (clipped_pos_z == np.array(state[2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in FlyThruGateAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(
                    state[2]
                ),
            )
        if not (clipped_rp == np.array(state[7:9])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in FlyThruGateAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                    state[7], state[8]
                ),
            )
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(
                    state[10], state[11]
                ),
            )
        if not (clipped_vel_z == np.array(state[12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in FlyThruGateAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(
                    state[12]
                ),
            )


class WaypointHandler:
    """Handler for Waypoints in the environments."""

    def __init__(
        self,
        enable_render: bool,
        num_targets: int,
        goal_reach_distance: float,
        goal_reach_angle: float,
        flight_dome_size: float,
        np_random: np.random.Generator,
    ):
        """__init__.

        Args:
            enable_render (bool): enable_render
            num_targets (int): num_targets
            goal_reach_distance (float): goal_reach_distance
            goal_reach_angle (float): goal_reach_angle
            flight_dome_size (float): flight_dome_size
            np_random (np.random.Generator): np_random
        """
        # constants
        self.enable_render = enable_render
        self.num_targets = num_targets
        self.goal_reach_distance = goal_reach_distance
        self.goal_reach_angle = goal_reach_angle
        self.flight_dome_size = flight_dome_size
        self.np_random = np_random

        # the target visual
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.targ_obj_dir = os.path.join(file_dir, "../../models/target.urdf")

    def reset(
        self,
        p,
        np_random: None | np.random.Generator = None,
    ):
        """Resets the waypoints."""
        # store the client
        self.p = p

        # update the random state
        if np_random is not None:
            self.np_random = np_random

        # reset the error
        self.new_distance = 0.0
        self.old_distance = 0.0

        # we sample from polar coordinates to generate linear targets
        self.targets = np.zeros(shape=(self.num_targets, 3))
        thetas = self.np_random.uniform(0.0, 2.0 * math.pi, size=(self.num_targets,))
        phis = self.np_random.uniform(0.0, 2.0 * math.pi, size=(self.num_targets,))
        for i, theta, phi in zip(range(self.num_targets), thetas, phis):
            dist = self.np_random.uniform(low=1.0, high=self.flight_dome_size * 0.9)
            x = dist * math.sin(phi) * math.cos(theta)
            y = dist * math.sin(phi) * math.sin(theta)
            z = abs(dist * math.cos(phi))

            # check for floor of z
            self.targets[i] = np.array([x, y, z if z > 0.1 else 0.1])


        # if we are rendering, load in the targets
        if self.enable_render:
            self.target_visual = []
            for target in self.targets:
                self.target_visual.append(
                    self.p.loadURDF(
                        self.targ_obj_dir,
                        basePosition=target,
                        useFixedBase=True,
                        globalScaling=self.goal_reach_distance / 4.0,
                    )
                )

            for i, visual in enumerate(self.target_visual):
                self.p.changeVisualShape(
                    visual,
                    linkIndex=-1,
                    rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                )

    def distance_to_target(
        self,
        lin_pos: np.ndarray,
        quarternion: np.ndarray,
    ):
        """distance_to_target.

        Args:
            ang_pos (np.ndarray): ang_pos
            lin_pos (np.ndarray): lin_pos
            quarternion (np.ndarray): quarternion
        """
        # rotation matrix
        rotation = np.array(self.p.getMatrixFromQuaternion(quarternion)).reshape(3, 3)

        # drone to target
        target_deltas = np.matmul((self.targets - lin_pos), rotation)

        # record distance to the next target
        self.old_distance = self.new_distance
        self.new_distance = float(np.linalg.norm(target_deltas[0]))

        return target_deltas

    def progress_to_target(self):
        """progress_to_target."""
        return self.old_distance - self.new_distance

    def target_reached(self):
        """target_reached."""
        if not self.new_distance < self.goal_reach_distance:
            return False

        return True

    def advance_targets(self):
        """advance_targets."""
        if len(self.targets) > 1:
            # still have targets to go
            self.targets = self.targets[1:]
        else:
            self.targets = []

        # delete the reached target and recolour the others
        if self.enable_render and len(self.target_visual) > 0:
            self.p.removeBody(self.target_visual[0])
            self.target_visual = self.target_visual[1:]

            # recolour
            for i, visual in enumerate(self.target_visual):
                self.p.changeVisualShape(
                    visual,
                    linkIndex=-1,
                    rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                )

    def num_targets_reached(self):
        """num_targets_reached."""
        return self.num_targets - len(self.targets)

    def all_targets_reached(self):
        """all_targets_reached."""
        return len(self.targets) == 0