from BaseRlControlAviary import WaypointsAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
    BaseSingleAgentAviary,
)
import numpy as np

def main():
    env = WaypointsAviary(gui=True, record=False, act=ActionType.ONE_D_DYN)
    
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        print(action.shape)
        print(np.array([-0.3750278]).shape)
        next_observations, reward, done, _ = env.step(np.array([0.1750278]))       
        print(f"Reward {reward}")
    input("Press Enter to continue...")
    env.close()

if __name__ == "__main__":
    main()
