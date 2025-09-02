"""
ALFWorld environment implementation for MAS evaluation.
"""
import os
import json
import numpy as np

from .alfworld_env import thor_env
from .alfworld_env.misc import get_templated_task_desc
from .alfworld_env.oracle import OracleAgent


class AlfredThorEnv(object):
    '''
    Interface for Embodied (THOR) environment
    '''

    def __init__(self, config):
        '''
        Initialize the environment.
        '''
        print("Initialize AlfredThorEnv for evaluation...")
        self.config = config
        self.env = None
        self.controller = None
        self.steps = 0
        self._done = False
        self._res = ()

        self.init_env()

    def init_env(self):
        '''
        Initialize the THOR environment.
        '''
        config = self.config
        print(f"DEBUG - environment.py - config: {config}")
        
        if not config:
            raise ValueError("Configuration is None or empty")
            
        if 'env' not in config:
            print(f"DEBUG - 'env' key missing in config")
            raise ValueError("'env' key missing in configuration")
            
        if 'thor' not in config.get('env', {}):
            print(f"DEBUG - 'thor' key missing in config['env']")
            raise ValueError("'thor' key missing in configuration['env']")
            
        try:
            screen_height = config['env']['thor']['screen_height']
            screen_width = config['env']['thor']['screen_width']
            smooth_nav = config['env']['thor']['smooth_nav']
            save_frames_to_disk = config['env']['thor']['save_frames_to_disk']
            print(f"DEBUG - thor config loaded: height={screen_height}, width={screen_width}")
        except Exception as e:
            print(f"DEBUG - Error accessing thor config: {e}")
            raise

        if not self.env:
            self.env = thor_env.ThorEnv(player_screen_height=screen_height,
                                        player_screen_width=screen_width,
                                        smooth_nav=smooth_nav,
                                        save_frames_to_disk=save_frames_to_disk)

    def reset(self, task_file):
        '''
        Reset the environment with a new task.
        '''
        print("Starting environment reset...")
        assert self.env, "Environment not initialized. Call init_env() first."

        try:
            # Load task data
            self.task_file = task_file
            self.traj_root = os.path.dirname(task_file)
            print(f"Loading task file: {task_file}")
            with open(task_file, 'r') as f:
                self.traj_data = json.load(f)
            print("Task file loaded successfully.")

            # Scene setup
            scene_num = self.traj_data['scene']['scene_num']
            object_poses = self.traj_data['scene']['object_poses']
            dirty_and_empty = self.traj_data['scene']['dirty_and_empty']
            object_toggles = self.traj_data['scene']['object_toggles']
            scene_name = 'FloorPlan%d' % scene_num
            
            print(f"Resetting scene: {scene_name}")
            self.env.reset(scene_name)
            print(f"Restoring scene...")
            self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)
            print("Scene restored.")

            # Initialize start position
            print("Initializing start position...")
            self.env.step(dict(self.traj_data['scene']['init_action']))
            print("Start position initialized.")

            # Print task description
            task_desc = get_templated_task_desc(self.traj_data)
            print(f"Task: {task_desc}")

            # Setup task for reward
            print("Setting up task for reward...")
            class args: pass
            args.reward_config = os.path.join(os.path.dirname(__file__), "alfworld_env", "config", "rewards.json")
            self.env.set_task(self.traj_data, args, reward_type='dense')
            print("Task setup for reward complete.")

            # Set controller
            print("Setting up controller...")
            self.setup_controller()
            print("Controller setup complete.")

            self.steps = 0
            self._done = False

            # Return initial feedback
            self._feedback = self.controller.feedback
            self._res = self.get_info()
            
            print("Environment reset finished.")
            return self._feedback, {}
        except Exception as e:
            import traceback
            print(f"An error occurred during environment reset: {e}")
            print(traceback.format_exc())
            # We can re-raise or handle it as needed. For debugging, let's return something to avoid crash
            return f"Error during reset: {e}", {"error": True}

    def setup_controller(self):
        '''
        Setup the controller based on config.
        '''
        controller_type = self.config['controller']['type']
        load_receps = self.config['controller']['load_receps']
        debug = self.config['controller']['debug']
        goal_desc_human_anns_prob = self.config['env']['goal_desc_human_anns_prob']

        if controller_type == 'oracle':
            self.controller = OracleAgent(self.env, self.traj_data, self.traj_root,
                                          load_receps=load_receps, debug=debug,
                                          goal_desc_human_anns_prob=goal_desc_human_anns_prob)
        else:
            raise NotImplementedError(f"Controller type '{controller_type}' not implemented. Only 'oracle' is currently supported after porting.")

    def step(self, action):
        '''
        Execute an action in the environment.
        '''
        if not self._done:
            self._feedback = self.controller.step(action)
            self._res = self.get_info()
        self.steps += 1

        obs, _, done, info = self.get_results()
        return obs, 0, done, info

    def get_results(self):
        '''
        Get the results of the last action.
        '''
        return self._res

    def get_info(self):
        '''
        Get information about the current state of the environment.
        '''
        won = self.env.get_goal_satisfied()
        pcs = self.env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1]) if pcs[1] > 0 else 0.0
        admissible_commands = self.controller.get_admissible_commands()

        max_steps = self.config.get('max_steps', 200)
        self._done = won or self.steps >= max_steps

        info = {
            'admissible_commands': admissible_commands,
            'won': won,
            'goal_condition_success_rate': goal_condition_success_rate,
            'gamefile': self.traj_root
        }

        return self._feedback, 0, self._done, info

    def close(self):
        '''
        Close the environment.
        '''
        if self.env:
            self.env.stop()

