from typing import Optional, List
import os
from enum import Enum
import numpy as np
import math
from gym import spaces
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
from gym_pybullet_drones.utils.utils import nnlsRPM
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl

from gym_pybullet_drones.envs.single_agent_rl.terminations import getTermDict
from gym_pybullet_drones.envs.single_agent_rl.terminations import BoundsTerm, OrientationTerm, CollisionTerm


class ActionType(Enum):
    """Action type enumeration class."""

    RPM = "rpm"  # RPMS
    DYN = "dyn"  # Desired thrust and torques
    PID = "pid"  # PID control
    VEL = "vel"  # Velocity input (using PID control)
    TUN = "tun"  # Tune the coefficients of a PID controller
    ONE_D_RPM = "one_d_rpm"  # 1D (identical input to all motors) with RPMs
    ONE_D_DYN = "one_d_dyn"  # 1D (identical input to all motors) with desired thrust and torques
    ONE_D_PID = "one_d_pid"  # 1D (identical input to all motors) with PID control


################################################################################


class ObservationType(Enum):
    """Observation type enumeration class."""

    KIN = "kin"  # Kinematic information (pose, linear and angular velocities)
    OBSTACLE = "obstacle"


################################################################################


class WaypointsAviary(BaseAviary):
    """Base single drone environment class for reinforcement learning."""

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
        tag: str = "",
        obstacle_list = [],
        num_waypoints: int = 5,
        goal_reach_distance: int = 0.2,
        bounds: List = [[5, 5, 5], [-5, -5, 0.1]],

    ):
        """Initialization of a generic single agent RL environment.

        Attribute `num_drones` is automatically set to 1; `vision_attributes`
        and `dynamics_attributes` are selected based on the choice of `obs`
        and `act`; `obstacles` is set to True and overridden with landmarks for
        vision applications; `user_debug_gui` is set to False for performance.

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
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        vision_attributes = False if (obs == ObservationType.KIN) else True
        dynamics_attributes = (
            True if act in [ActionType.DYN, ActionType.ONE_D_DYN] else False
        )
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = 5
        self._seed = (
            0  # bullet is deterministic, but we set seed for algorithms which need it
        )
        self.num_waypoints = num_waypoints
        self.goal_reach_distance = goal_reach_distance
        self.bounds = bounds

        self.term_components = []
        if len(self.term_components) == 0:
            self.term_components.append(BoundsTerm(bounds=self.bounds))
            self.term_components.append(OrientationTerm())
        self.term_dict = getTermDict(self.term_components)
        self.truncated = False
        self.done = False

        self.completeEpisode = False
        self.min_dist = 100
        #### Create integrated controllers #########################
        if act in [
            ActionType.PID,
            ActionType.VEL,
            ActionType.TUN,
            ActionType.ONE_D_PID,
        ]:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
                if act == ActionType.TUN:
                    self.TUNED_P_POS = np.array([0.4, 0.4, 1.25])
                    self.TUNED_I_POS = np.array([0.05, 0.05, 0.05])
                    self.TUNED_D_POS = np.array([0.2, 0.2, 0.5])
                    self.TUNED_P_ATT = np.array([70000.0, 70000.0, 60000.0])
                    self.TUNED_I_ATT = np.array([0.0, 0.0, 500.0])
                    self.TUNED_D_ATT = np.array([20000.0, 20000.0, 12000.0])
            elif drone_model == DroneModel.HB:
                self.ctrl = SimplePIDControl(drone_model=DroneModel.HB)
                if act == ActionType.TUN:
                    self.TUNED_P_POS = np.array([0.1, 0.1, 0.2])
                    self.TUNED_I_POS = np.array([0.0001, 0.0001, 0.0001])
                    self.TUNED_D_POS = np.array([0.3, 0.3, 0.4])
                    self.TUNED_P_ATT = np.array([0.3, 0.3, 0.05])
                    self.TUNED_I_ATT = np.array([0.0001, 0.0001, 0.0001])
                    self.TUNED_D_ATT = np.array([0.3, 0.3, 0.5])
            else:
                print(
                    "[ERROR] in BaseSingleAgentAviary.__init()__, no controller is available for the specified drone_model"
                )
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record,
            obstacles=True,  # Add obstacles for RGB observations and/or FlyThruGate
            user_debug_gui=False,  # Remove of RPM sliders from all single agent learning aviaries
            vision_attributes=vision_attributes,
            dynamics_attributes=dynamics_attributes,
            tag=tag,
        )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000 / 3600)
        #### Try _trajectoryTrackingRPMs exists IFF ActionType.TUN #
        if act == ActionType.TUN and not (
            hasattr(self.__class__, "_trajectoryTrackingRPMs")
            and callable(getattr(self.__class__, "_trajectoryTrackingRPMs"))
        ):
            print(
                "[ERROR] in BaseSingleAgentAviary.__init__(), ActionType.TUN requires an implementation of _trajectoryTrackingRPMs in the instantiated subclass"
            )
            exit()

    def seed(self, seed: int):
        self._seed = seed

    ################################################################################
    def getFirstDroneState(self):
        return self._getDroneStateVector(0)

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        for obstacle in self.obstacles:
            obstacle._addObstacles()
        # we sample from polar coordinates to generate linear targets
        self.targets = np.zeros(shape=(self.num_waypoints, 3))
        thetas = np.random.uniform(0.0, 2.0 * math.pi, size=(self.num_waypoints,))
        phis = np.random.uniform(0.0, 2.0 * math.pi, size=(self.num_waypoints,))
        for i, theta, phi in zip(range(self.num_waypoints), thetas, phis):
            dist = np.random.uniform(low=1.0, high=5 * 0.9)
            x = dist * math.sin(phi) * math.cos(theta)
            y = dist * math.sin(phi) * math.sin(theta)
            z = abs(dist * math.cos(phi))

            # check for floor of z
            self.targets[i] = np.array([x, y, z if z > 0.1 else 0.1])

        self.new_distance = 0.0
        self.old_distance = 0.0

        self.target_visual = []
        for target in self.targets:
            self.target_visual.append(
                p.loadURDF(
                    os.path.dirname(os.path.abspath(__file__))
                    + "/../../assets/waypoint.urdf",
                    basePosition=target,
                    useFixedBase=True,
                    globalScaling=0.05,
                )
            )

        for i, visual in enumerate(self.target_visual):
            p.changeVisualShape(
                visual,
                linkIndex=-1,
                rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
            )

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        ndarray
            A Box() of size 1, 3, 4, or 6 depending on the action type.

        """
        if self.ACT_TYPE == ActionType.TUN:
            size = 6
        elif self.ACT_TYPE in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE == ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [
            ActionType.ONE_D_RPM,
            ActionType.ONE_D_DYN,
            ActionType.ONE_D_PID,
        ]:
            size = 1
        else:
            print("[ERROR] in BaseSingleAgentAviary._actionSpace()")
            exit()
        return spaces.Box(
            low=-1 * np.ones(size),
            # return spaces.Box(low=np.zeros(size),  # Alternative action space, see PR #32
            high=np.ones(size),
            dtype=np.float32,
        )

    ################################################################################

    def _preprocessAction(self, action):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, 4, or 6 and represent
        RPMs, desired thrust and torques, the next target position to reach
        using PID control, a desired velocity vector, new PID coefficients, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (4,)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        if self.ACT_TYPE == ActionType.TUN:
            self.ctrl.setPIDCoefficients(
                p_coeff_pos=(action[0] + 1) * self.TUNED_P_POS,
                i_coeff_pos=(action[1] + 1) * self.TUNED_I_POS,
                d_coeff_pos=(action[2] + 1) * self.TUNED_D_POS,
                p_coeff_att=(action[3] + 1) * self.TUNED_P_ATT,
                i_coeff_att=(action[4] + 1) * self.TUNED_I_ATT,
                d_coeff_att=(action[5] + 1) * self.TUNED_D_ATT,
            )
            return self._trajectoryTrackingRPMs()
        elif self.ACT_TYPE == ActionType.RPM:
            return np.array(self.HOVER_RPM * (1 + 0.05 * action))
        elif self.ACT_TYPE == ActionType.DYN:
            return nnlsRPM(
                thrust=(self.GRAVITY * (action[0] + 1)),
                x_torque=(0.05 * self.MAX_XY_TORQUE * action[1]),
                y_torque=(0.05 * self.MAX_XY_TORQUE * action[2]),
                z_torque=(0.05 * self.MAX_Z_TORQUE * action[3]),
                counter=self.step_counter,
                max_thrust=self.MAX_THRUST,
                max_xy_torque=self.MAX_XY_TORQUE,
                max_z_torque=self.MAX_Z_TORQUE,
                a=self.A,
                inv_a=self.INV_A,
                b_coeff=self.B_COEFF,
                gui=self.GUI,
            )
        elif self.ACT_TYPE == ActionType.PID:
            state = self._getDroneStateVector(0)
            rpm, _, _ = self.ctrl.computeControl(
                control_timestep=self.AGGR_PHY_STEPS * self.TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=state[0:3] + 0.1 * action,
            )
            return rpm
        elif self.ACT_TYPE == ActionType.VEL:
            state = self._getDroneStateVector(0)
            if np.linalg.norm(action[0:3]) != 0:
                v_unit_vector = action[0:3] / np.linalg.norm(action[0:3])
            else:
                v_unit_vector = np.zeros(3)
            rpm, _, _ = self.ctrl.computeControl(
                control_timestep=self.AGGR_PHY_STEPS * self.TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=state[0:3],  # same as the current position
                target_rpy=np.array([0, 0, state[9]]),  # keep current yaw
                target_vel=self.SPEED_LIMIT
                * np.abs(action[3])
                * v_unit_vector,  # target the desired velocity vector
            )
            return rpm
        elif self.ACT_TYPE == ActionType.ONE_D_RPM:
            return np.repeat(self.HOVER_RPM * (1 + 0.05 * action), 4)
        elif self.ACT_TYPE == ActionType.ONE_D_DYN:
            return nnlsRPM(
                thrust=(self.GRAVITY * (1 + 0.05 * action[0])),
                x_torque=0,
                y_torque=0,
                z_torque=0,
                counter=self.step_counter,
                max_thrust=self.MAX_THRUST,
                max_xy_torque=self.MAX_XY_TORQUE,
                max_z_torque=self.MAX_Z_TORQUE,
                a=self.A,
                inv_a=self.INV_A,
                b_coeff=self.B_COEFF,
                gui=self.GUI,
            )
        elif self.ACT_TYPE == ActionType.ONE_D_PID:
            state = self._getDroneStateVector(0)
            rpm, _, _ = self.ctrl.computeControl(
                control_timestep=self.AGGR_PHY_STEPS * self.TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=state[0:3] + 0.1 * np.array([0, 0, action[0]]),
            )
            return rpm
        else:
            print("[ERROR] in BaseSingleAgentAviary._preprocessAction()")

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
            # obs_lower_bound = np.array([-1,      -1,      0,      -1,  -1,  -1,  -1,  -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1,      -1,           -1,           -1,           -1])
            # obs_upper_bound = np.array([1,       1,       1,      1,   1,   1,   1,   1,      1,      1,      1,       1,       1,       1,       1,       1,       1,            1,            1,            1])
            # return spaces.Box( low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32 )
            ############################################################
            #### OBS SPACE OF SIZE 12
            return spaces.Box(
                low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
                high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                dtype=np.float32,
            )
            ############################################################
        elif self.OBS_TYPE == ObservationType.OBSTACLE:

            return spaces.Dict(
                spaces={
                    "vec": spaces.Box(
                        low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
                        high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                        dtype=np.float32,
                    ),
                    "target": spaces.Box(
                        low=np.array([-2 * 5, -2 * 5, -2 * 5]),
                        high=np.array([2 * 5, 2 * 5, 2 * 5]),
                        dtype=np.float32,
                    )
                }
            )
        else:
            print("[ERROR] in BaseSingleAgentAviary._observationSpace()")

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.KIN:
            obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            # return obs
            ############################################################
            #### OBS SPACE OF SIZE 12
            ret = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            return ret.astype("float32")
            ############################################################
        elif self.OBS_TYPE == ObservationType.OBSTACLE:
            obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            # return obs
            ############################################################
            #### OBS SPACE OF SIZE 12
            ret = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            dist = self._distance_to_target(obs[0:3], obs[3:7])[0]
            self.distance_to_immediate = float(
                np.linalg.norm(dist)
            )
            return {"vec": ret.astype("float32"), "target": dist.astype("float32")}
            ############################################################
        else:
            print("[ERROR] in BaseSingleAgentAviary._computeObs()")

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
                "in WaypointAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(
                    state[0], state[1]
                ),
            )
        if not (clipped_pos_z == np.array(state[2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in WaypointAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(
                    state[2]
                ),
            )
        if not (clipped_rp == np.array(state[7:9])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in WaypointAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                    state[7], state[8]
                ),
            )
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in WaypointAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(
                    state[10], state[11]
                ),
            )
        if not (clipped_vel_z == np.array(state[12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in WaypointAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(
                    state[12]
                ),
            )
    
    def _distance_to_target(
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
        rotation = np.array(p.getMatrixFromQuaternion(quarternion)).reshape(3, 3)

        # drone to target
        target_deltas = np.matmul((self.targets - lin_pos), rotation)

        # record distance to the next target
        self.old_distance = self.new_distance
        self.new_distance = float(np.linalg.norm(target_deltas[0]))

        return target_deltas
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {
            "targets reached": self.num_targets - len(self.targets)
        } 
    
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
            self.done = done
        self.done = self.done or len(self.targets) == 0
        return self.done
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        reward = 0
        reward += max(3.0 * (self.old_distance - self.new_distance), 0.0)
        reward += 0.1 / self.distance_to_immediate

        if self.new_distance < self.goal_reach_distance:
            reward = 100.0

            # advance the targets
            self.advance_targets()

        return reward
    
    def advance_targets(self):
        """advance_targets."""
        if len(self.targets) > 1:
            # still have targets to go
            self.targets = self.targets[1:]
        else:
            self.targets = []

        # delete the reached target and recolour the others
        if len(self.target_visual) > 0:
            self.p.removeBody(self.target_visual[0])
            self.target_visual = self.target_visual[1:]

            # recolour
            for i, visual in enumerate(self.target_visual):
                self.p.changeVisualShape(
                    visual,
                    linkIndex=-1,
                    rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                )


    

        
