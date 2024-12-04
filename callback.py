from copy import deepcopy
import time
import os
import wandb
import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class CustomEvalCallback(BaseCallback):
    """
    A custom callback for evaluating a model at regular intervals during training.
    
    :param eval_env: The environment used for evaluation
    :param eval_freq: Frequency (in steps) at which the evaluation is performed
    :param save_freq: Frequency (in steps) at which the model is saved
    :param n_eval_episodes: Number of episodes to run during each evaluation (default is 5)
    :param best_model_save_path: Directory to save the best model (default is "test/model")
    :param deterministic: Whether to use deterministic actions during evaluation (default is True)
    :param log_wandb: Flag to log results to Weights & Biases (WandB) (default is True)
    :param sim: Whether this is a simulation environment (default is False)
    :param verbose: Verbosity level (0: no output, 1: info, 2: debug)
    """

    def __init__(
        self,
        eval_env,
        eval_freq,
        save_freq,
        n_eval_episodes=5,
        best_model_save_path=f"test/model",
        deterministic=True,
        log_wandb=True,
        sim=False,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        # Evaluation environment and configuration
        self.eval_env = eval_env
        self.eval_freq = eval_freq  # Frequency of evaluation
        self.save_freq = save_freq  # Frequency of saving the model
        self.n_eval_episodes = n_eval_episodes  # Number of episodes for evaluation
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.log_wandb = log_wandb  # Whether to log results to WandB
        self.sim = sim  # Whether this is a simulation environment

        # Variables for tracking best results during evaluation
        self.best_mean_reward = -np.inf
        self.eval_best_config = {}
        self.eval_best_step = 0
        self.eval_best_totaltime = 0
        self.eval_best_compiletime = 0

        # Variables for tracking best results during training
        self.train_best_config = {}
        self.train_best_step = 0
        self.train_best_reward = -np.inf
        self.train_best_timeflag = 0

        # Timing and environment settings
        self.start_time = None
        self.top_k = 5  # Number of top configurations to keep track of
        self.actions = [
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_N",
            "BLOCK_SIZE_K",
            "num_stages",
            "num_warps",
        ]
        self.datasizes = deepcopy(self.eval_env.get_datasizes())  # Clone environment datasizes

    def _init_callback(self):
        """
        This method is called once at the start of the training process.
        Used for custom initialization.
        """
        pass

    def _on_training_start(self) -> None:
        """
        Called before the first training rollout.
        Sets the start time for timing evaluations.
        """
        self.start_time = time.time()

    def _on_step(self) -> bool:
        """
        Called at each step during the training process.
        Performs evaluation and logging at predefined intervals.
        
        :return: True to continue training, False to stop early.
        """
        # Evaluate the model at regular intervals
        if self.n_calls % self.eval_freq == 0:
            self.evaluate_kernel()

        # Save the model at regular intervals
        if self.n_calls % self.save_freq == 0:
            self.save_progress_report()

        return True

    def evaluate_kernel(self):
        """
        Evaluates the model by running several episodes in the evaluation environment.
        Logs results and saves the best model configuration if performance improves.
        """
        episode_rewards = []  # List to store rewards per episode
        eval_best = {size: {"latency": 100, "info": {}} for size in self.datasizes}  # Initialize best results for each data size

        # Run multiple evaluation episodes
        for _ in range(self.n_eval_episodes):
            terminate = False
            obs, _ = self.eval_env.reset()  # Reset the evaluation environment
            total_reward = 0.0

            # Run the evaluation until the episode ends
            while not terminate:
                action, _states = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminate, _, info = self.eval_env.step(action)
                size = info["size"]
                # Update the best configuration if the performance improves
                if info["latency"] < eval_best[size]["latency"]:
                    eval_best[size]["latency"] = info["latency"]
                    eval_best[size]["info"] = info

                total_reward += reward  # Accumulate the reward for the episode

            episode_rewards.append(total_reward)  # Store the total reward for this episode

        # Calculate the mean reward across all evaluation episodes
        mean_reward = sum(episode_rewards) / self.n_eval_episodes

        # Prepare data for logging (WandB or other)
        eval_data = {
            "time/compile_time": self.training_env.get_attr("total_compile_time")[0],
            "time/steps": self.num_timesteps,
            # "time/total_time": self.training_env.get_attr("total_time")[0],
            "eval/mean_reward": mean_reward,
        }

        # Calculate and log the standard deviation of each action's performance
        for action in self.actions:
            values = [result["info"][action] for result in eval_best.values()]
            eval_data[f"eval/{action}_std"] = np.std(values)

        # Log the evaluation results to WandB if enabled
        if self.log_wandb:
            wandb.log(eval_data, step=self.num_timesteps)

        # Check if the current evaluation resulted in a new best mean reward
        if mean_reward > self.best_mean_reward:
            # Save the best evaluation results
            self.eval_best_totaltime = time.time() - self.start_time
            self.eval_best_compiletime = self.training_env.get_attr("total_compile_time")[0]
            self.eval_best_step = self.num_timesteps
            self.eval_best_config = eval_best

            # Save the model if a save path is provided
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))

            self.best_mean_reward = float(mean_reward)  # Update the best mean reward
            print("New best evaluation reward!")
            print("Eval reward:", mean_reward)

    def save_progress_report(self) -> None:
        """
        Saves a report of the current training progress, including timing and configuration information.
        """
        exp_folder = self.training_env.get_attr("exp_folder")[0]
        gpu = self.training_env.get_attr("device")[0]
        elapsed_time = self.training_env.get_attr("total_compile_time")[0]

        # Create a JSON report with training and evaluation details
        report_data = {
            "general": {
                "GPU": gpu,
                "Total_time": self.format_time(elapsed_time),
            },
            "eval": {
                "elapsed_time": self.format_time(self.eval_best_totaltime),
                "compile_time": self.format_time(self.eval_best_compiletime),
                "best_step": self.eval_best_step,
                "best_config": self.eval_best_config,
            },
        }

        # Save the report to a JSON file
        with open(os.path.join(exp_folder, f"report_{self.num_timesteps}.json"), "w") as file:
            json.dump(report_data, file, indent=4)

    def _on_training_end(self) -> None:
        """
        Called when training ends, generates a final report of the best training and evaluation results.
        """
        exp_folder = self.training_env.get_attr("exp_folder")[0]
        gpu = self.training_env.get_attr("device")[0]
        failed_configs = self.training_env.get_attr("failed_configs")[0]

        # Determine elapsed time and frames per second (FPS)
        elapsed_time = time.time() - self.start_time
        fps = self.num_timesteps / elapsed_time

        # Save the results of the best configurations and failures to JSON
        results = self.training_env.get_attr("benchmark_results")[0]
        sorted_results = sorted(results, key=lambda x: x["latency"])

        with open(os.path.join(exp_folder, f"datas.json"), "w") as file:
            json.dump(sorted_results, file, indent=4)

        if failed_configs:
            failed_configs.insert(0, {"Fail count": len(failed_configs)})
            with open(os.path.join(exp_folder, f"fails.json"), "w") as file:
                json.dump(failed_configs, file, indent=4)

        # Generate a final report of the training process
        final_report = {
            "general": {
                "GPU": gpu,
                "Total_time": self.format_time(elapsed_time),
                "FPS": fps,
                "Fail_count": len(failed_configs),
            },
            # "train": {
            #     "elapsed_time": self.format_time(self.train_best_timeflag),
            #     "best_step": self.train_best_step,
            #     "best_config": self.train_best_config,
            # },
            "eval": {
                "elapsed_time": self.format_time(self.eval_best_totaltime),
                "compile_time": self.format_time(self.eval_best_compiletime),
                "best_step": self.eval_best_step,
                "best_config": self.eval_best_config,
            },
                }

        # Save the final training report to a JSON file
        with open(os.path.join(exp_folder, f"report.json"), "w") as file:
            json.dump(final_report, file, indent=4)

        # Print final results to the console
        print("Total runtime:", self.format_time(elapsed_time))
        print("------------- Best Training Config -------------")
        print("Best Training Step:", self.train_best_step)
        print(self.train_best_config)
        print("------------- Best Evaluation Config -------------")
        print("Best Evaluation Step:", self.eval_best_step)
        print(self.eval_best_config)

    def format_time(self, elapsed_time: float) -> str:
        """
        Utility function to format elapsed time into hours, minutes, and seconds.

        :param elapsed_time: Time in seconds
        :return: Formatted time string in the format HH:MM:SS
        """
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"