import torch
import triton
import triton.language as tl
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import json
import os
import time
import signal
import random
import signal

# Custom exception for timeout functionality
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


# Triton kernel for matrix multiplication
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,  # Pointers to matrices A, B, C
    M, N, K,  # Matrix dimensions
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  # Strides for matrices A, B, C
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize an accumulator for the block of C in FP32 for accuracy
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, use a mask if out of bounds
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # Accumulate the result in the accumulator 
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply activation function if specified
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

# Function to perform matrix multiplication using Triton kernel
def matmul(a, b, config, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        ACTIVATION=activation,  #
        BLOCK_SIZE_M=config[0],
        BLOCK_SIZE_N=config[1],
        BLOCK_SIZE_K=config[2],
        GROUP_SIZE_M=8,
        num_stages=config[3],
        num_warps=config[4],
    )
    return c


# Custom gymnasium environment for RL-based matrix multiplication optimization
class MatmulEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    # metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        config,
        exp_folder="data/test",
        device=0,
        mode="train",
        random_sample="False",
    ):
        """Initialize the environment with the provided configuration.

        Args:
            config (dict): Configuration dictionary containing various environment parameters.
            exp_folder (str): Path to store experimental data (default: "data/test").
            device (int): GPU device index (default:0).
            mode (str): Mode of the environment, either 'train' or 'search' (default: "train").
            random_sample (bool): Whether to randomly sample data sizes or iterate through them (default: False).
        """
        super().__init__()

        # Initialize configuration parameters
        self.cudagraph_enabled = config["cudagraph"]   # Whether CUDA graph optimization is enabled
        self.episode_len = config["episode_len"]       # Length of each episode
        self.simulation_mode = config["sim"]           # Whether in simulation mode or real benchmark mode
        self.fp_precision = config["fp"]               # Floating point precision (e.g., 16-bit or 8-bit)
        self.size = config["size"]                     # Size of the matrix for benchmarking
        self.warmup = config["warmup"]                 # Milliseconds of warmup runs for the kernel benchmark
        self.repeat = config["repeat"]                 # Milliseconds of benchmark repetitions for accurate timing
        self.search_space = config["search_space"]     # Search space for kernel tuning parameters
        self.datasizes = config["datasizes"]           # List of matrix sizes to iterate through

        # Dictionary to store benchmark results and compile times
        self.latency_dict = {}     # Store kernel latency (ms)
        self.compile_time_dict = {} # Store kernel compilation time

        # Load data if in simulation mode
        if self.simulation_mode:
            self.load_simulation_data()
        
        # Action and observation space definitions
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)   # 5 tuning parameters
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # 3 matrix dimensions

        # Experiment folder and device settings
        self.exp_folder = exp_folder
        self.device = device

        # Initial matrix sizes for the environment
        self.current_M = self.datasizes[0]
        self.current_N = self.datasizes[0]
        self.current_K = self.datasizes[0]

        self.mode = mode
        self.random_sample = random_sample

        # If in search mode, generate all possible configurations to search over
        self.configs = []
        if self.mode == "search":
            for M in self.search_space[0]:
                for N in self.search_space[1]:
                    for K in self.search_space[2]:
                        for stages in self.search_space[3]:
                            for warps in self.search_space[4]:
                                self.configs.append([M, N, K, stages, warps])

        # Initialize benchmarking metrics and counters
        self.steps = 0
        self.failures = 0
        self.total_compile_time = 0
        self.best_config = []
        self.benchmark_results = []
        self.failed_configs = []
        self.visited_configs = []
        self.start_time = time.time()


        self.action_values = self.search_space
        print("Search Space")
        print(self.search_space)

        # Set the GPU device
        torch.cuda.set_device(device)
        print("Available GPUs: ", torch.cuda.device_count())
        print("Current GPU: ", torch.cuda.current_device())
        # print("GPU Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

    def load_simulation_data(self):
        """Load precomputed simulation data from JSON files."""
        data_folder = f"./simdata/FP{self.fp_precision}-R{self.repeat}"
        
        # Load kernel execution data for different matrix sizes
        for size in self.datasizes:
            result_file = os.path.join(data_folder, f"{size}_results.json")
            with open(result_file, "r") as f:
                data = json.load(f)
                for config in data:
                    key = (size, config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"], config["BLOCK_SIZE_K"], config["num_stages"], config["num_warps"])
                    self.latency_dict[key] = config["ms"]
                    self.compile_time_dict[key] = config["compile_time"]
            
            # Load kernel failure data
            fail_file = os.path.join(data_folder, f"{size}_fails.json")
            with open(fail_file, "r") as f:
                data = json.load(f)
                for config in data:
                    key = (size, config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"], config["BLOCK_SIZE_K"], config["num_stages"], config["num_warps"])
                    self.latency_dict[key] = -1  # Indicating a failed configuration
                    self.compile_time_dict[key] = config["compile_time"]

    def simulate_benchmark(self, size, actions):
        """Simulate benchmarking in simulation mode using precomputed results.

        Args:
            size (int): Matrix size for benchmarking.
            actions (list): List of kernel parameters [BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps].
        
        Returns:
            tuple: Reward and latency (ms) for the given configuration.
        """
        # Retrieve precollected latency and compilation time
        latency = self.latency_dict.get((size, actions[0], actions[1], actions[2], actions[3], actions[4]))
        compile_time = self.compile_time_dict.get((size, actions[0], actions[1], actions[2], actions[3], actions[4]))

        config = {
            "size": size,
            "BLOCK_SIZE_M": actions[0],
            "BLOCK_SIZE_N": actions[1],
            "BLOCK_SIZE_K": actions[2],
            "GROUP_SIZE_M": 8,
            "num_stages": actions[3],
            "num_warps": actions[4],
            "compile_time": compile_time,
        }

        # If latency is -1, the configuration failed
        if latency == -1:
            self.failures += 1
            if config in self.visited_configs:
                reward = -100  # Heavy penalty for retrying a failed config
            else:
                reward = -10
                self.visited_configs.append(config)
                self.total_compile_time += compile_time
        else:
            # Compute performance in terms of FLOPS (Floating-point Operations per Second)
            performance = lambda ms: 2 * size * size * size * 1e-12 / (ms * 1e-3)
            reward = performance(latency)
            if config not in self.visited_configs:
                self.visited_configs.append(config)
                self.total_compile_time += compile_time
        
        # self.total_compile_time += compile_time
        return reward, latency

    def true_benchmark(self, size, a, b, actions):
        """Run the actual benchmark using Triton and record the results.

        Args:
            size (int): Matrix size for benchmarking.
            a (torch.Tensor): Input matrix A.
            b (torch.Tensor): Input matrix B.
            actions (list): List of kernel parameters [BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps].
        
        Returns:
            tuple: Reward and latency (ms) for the given configuration.
        """
        key = (
            size,
            actions[0],  # BLOCK_SIZE_M
            actions[1],  # BLOCK_SIZE_N
            actions[2],  # BLOCK_SIZE_K
            actions[3],  # num_stages (pipeline stages)
            actions[4],  # num_warps (number of warps for parallelism)
        )

        # Check if the results for this configuration have been cached (i.e., previously computed)
        latency = self.latency_dict.get(key)
        compile_time = self.compile_time_dict.get(key)
        # If this configuration has not been run before, proceed with benchmarking
        if latency is None:
            try:
                # Set a timeout handler in case the benchmark takes too long
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(6)  # Set the alarm for 6 seconds

                compile_start = time.time()
                
                # Benchmark using CUDA graphs if enabled, otherwise perform regular benchmarking
                if self.cudagraph_enabled:
                    # CUDA Stream allows asynchronous operations on the GPU
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        torch.cuda.synchronize() # Ensure all previous GPU tasks are finished
                        latency = triton.testing.do_bench_cudagraph(lambda: matmul(a, b, actions), rep=self.repeat)
                    min_latency = max_latency = latency # No min/max quantiles for cudagraph
                else:
                    # Run the Triton benchmark and calculate latencies for different quantiles
                    latency, min_latency, max_latency = triton.testing.do_bench(
                        lambda: matmul(a, b, actions),
                        quantiles=[0.5, 0.2, 0.8],
                        warmup=self.warmup,
                        rep=self.repeat,
                    )
                    
                compile_time = time.time() - compile_start
                # print("Compilation completed successfully in:", compile_time, "seconds")
                signal.alarm(0) # Reset the alarm

                # Create a dictionary to store the successful configuration results
                success_config = {
                    "BLOCK_SIZE_M": actions[0],
                    "BLOCK_SIZE_N": actions[1],
                    "BLOCK_SIZE_K": actions[2],
                    "GROUP_SIZE_M": 8,
                    "num_stages": actions[3],
                    "num_warps": actions[4],
                    "latency": latency,
                    "min_latency": min_latency,
                    "max_latency": max_latency,
                    "compile_time": compile_time,
                }
                self.benchmark_results.append(success_config) # Store the result

                # Calculate performance (TFLOPs) based on latency (2 * M * N * K / latency in seconds)
                performance = lambda latency: 2 * self.current_M * self.current_N * self.current_K * 1e-12 / (latency * 1e-3)
                reward = performance(latency) # Reward is the performance metric in TFLOPs
                
                # Cache the results for future runs of this configuration
                self.latency_dict[key] = latency
                self.compile_time_dict[key] = compile_time
            except (TimeoutException, Exception) as e:
                # Handle errors such as timeouts or failures during compilation
                print("Error occurred during benchmarking:", e)
                compile_time = time.time() - compile_start
                print("Failed compilation time:", compile_time)
                ### DEBUG
                # print(
                #     f"BLOCK_SIZE_M= {actions[0]}, BLOCK_SIZE_N= {actions[1]}, BLOCK_SIZE_K= {actions[2]}, GROUP_SIZE_M= 8, num_stages={actions[3]}, num_warps={actions[4]}"
                # )

                # Store the failed configuration in the failed list
                failed_config = {
                    "BLOCK_SIZE_M": actions[0],
                    "BLOCK_SIZE_N": actions[1],
                    "BLOCK_SIZE_K": actions[2],
                    "GROUP_SIZE_M": 8,
                    "num_stages": actions[3],
                    "num_warps": actions[4],
                    "compile_time": compile_time,
                }
                self.failed_configs.append(failed_config)

                # Mark latency as failed (-1) and assign a negative reward
                latency = -1
                reward = -10
                self.failures += 1
                self.latency_dict[key] = -1
                self.compile_time_dict[key] = compile_time
        else:
            # If the configuration has been seen before, return the cached result
            print("Configuration has been seen before!")
            if latency == -1:
                self.failures += 1
                reward = -100 # Assign a large negative reward for failed configurations
            else:
                # If successful, calculate the reward based on performance (TFLOPs)
                performance = lambda latency: 2 * size * size * size * 1e-12 / (latency * 1e-3)
                reward = performance(latency)

        self.total_compile_time += compile_time
        return reward, latency

    def step(self, action):
        """Perform a step in the environment based on the agent's action.

        Args:
            action (list): The actions chosen by the agent, corresponding to block size and other parameters.

        Returns:
            tuple: observation (normalized M, N, K sizes), reward (performance metric), 
            terminated (whether the episode ended), truncated (whether the episode was truncated), 
            info (dictionary containing benchmarking details).
        """
        actions = []

        # Use predefined configurations in search mode, otherwise derive the configuration from the action.
        if self.mode == "search":
            actions = self.configs[self.steps % len(self.configs)] # Cycle through configurations
        else:
            # Map the continuous action space values to actual discrete parameters from the search space
            for index, act in enumerate(action):
                normalized_action = (act + 1) / 2 # Normalize action from [-1, 1] to [0, 1]
                key = round(normalized_action * (len(self.action_values[index]) - 1)) # Convert to search space index
                actions.append(self.action_values[index][key]) # Get actual search space value

        ### DEBUG
        # print(
        #     f"BLOCK_SIZE_M= {actions[0]}, BLOCK_SIZE_N= {actions[1]}, BLOCK_SIZE_K= {actions[2]}, GROUP_SIZE_M= 8, num_stages={actions[3]}, num_warps={actions[4]}"
        # )

        # Simulate or perform a true benchmark depending on the environment configuration
        if self.simulation_mode:
            reward, latency = self.simulate_benchmark(self.current_M, actions)
        else:
            # Create random input matrices a and b for matrix multiplication
            a = torch.randn((self.current_M, self.current_K), device="cuda", dtype=torch.float16)
            b = torch.randn((self.current_K, self.current_N), device="cuda", dtype=torch.float16)
            #
            #  If using FP8 precision, convert the matrices to FP8
            if self.fp_precision == 8:
                a = a.to(torch.float8_e5m2)
                b = b.T.to(torch.float8_e5m2) # Pre-transpose b for efficiency

            reward, latency = self.true_benchmark(self.current_M, a, b, actions)

        self.steps += 1

        # Determine if the episode has ended after reaching the episode length
        terminated = False
        if self.steps % self.episode_len == 0:
            terminated = True

        # Info dictionary with benchmarking details for the current step
        info = {
            "size": self.current_M,
            "latency": latency,
            "BLOCK_SIZE_M": actions[0],
            "BLOCK_SIZE_N": actions[1],
            "BLOCK_SIZE_K": actions[2],
            "num_stages": actions[3],
            "num_warps": actions[4],
        }

        # Randomly sample a new matrix size (M, N, K) for the next step
        if self.random_sample:
            index = random.randint(0, len(self.datasizes) - 1)
        else:
            index = self.steps % len(self.datasizes)

        self.current_M = self.current_N = self.current_K = self.datasizes[index] # Update M, N, K for the next step
        # observation = np.array([self.current_M, self.current_N, self.current_K], dtype=np.uint16)
        # Normalized observation (M, N, K) for the agent, scaled between [-1, 1]
        observation = np.array(
            [self.current_M / 5000 - 1, self.current_N / 5000 - 1, self.current_K / 5000 - 1], dtype=np.float32
        )

        # Handle cases where the benchmark failed (latency = -1)
        if latency == -1:
            latency = 10 # Arbitrary high value for failed cases


        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        """Reset the environment at the start of a new episode.

        Args:
            seed (int, optional): Seed for reproducibility.
            options (dict, optional): Additional options for resetting.

        Returns:
            tuple: observation (normalized M, N, K sizes), and info (empty in this case).
        """
        # Randomly sample a new matrix size (M, N, K) for the new episode
        if self.random_sample:
            index = random.randint(0, len(self.datasizes) - 1)
        else:
            self.steps = 0
            # random.shuffle(self.datasizes)
            index = self.steps % len(self.datasizes)
        self.current_M = self.current_N = self.current_K = self.datasizes[index]

        # Normalized observation (M, N, K) for the agent, scaled between [-1, 1]
        observation = np.array(
            [self.current_M / 5000 - 1, self.current_N / 5000 - 1, self.current_K / 5000 - 1], dtype=np.float32
        )
        info = {}
        return observation, info

    def report(self, fps, top_k=5):
        # if self.mode != "search":
        #     print("Only Search mode support this function")
        #     return
        # Sorting by 'ms'
        total_time = time.time() - self.start_time
        sorted_by_ms = sorted(self.benchmark_results, key=lambda x: x["latency"])
        sorted_by_ms.insert(0, {"FPS": fps, "GPU": self.device, "runtime": total_time})
        for i in range(top_k):
            print(sorted_by_ms[i])
        # Open the file in write mode and save the dictionary
        with open(os.path.join(self.exp_folder, f"results.json"), "w") as file:
            json.dump(
                sorted_by_ms, file, indent=4
            )  # 'indent' is optional, it makes the file more readable

        print("Failed config counts", len(self.failed_configs))
        if len(self.failed_configs):
            self.failed_configs.insert(0, {"Fail count": len(self.failed_configs)})
            with open(os.path.join(self.exp_folder, f"fails.json"), "w") as file:
                json.dump(
                    self.failed_configs, file, indent=4
                )  # 'indent' is optional, it makes the file more readable

    def cublas(self):
        a = torch.randn((self.current_M, self.current_K), device="cuda", dtype=torch.float16)
        b = torch.randn((self.current_K, self.current_N), device="cuda", dtype=torch.float16)
        if self.fp_precision == 8:
            a = a.to(torch.float8_e5m2)
            # pre-transpose b for efficiency.
            b = b.T
            b = b.to(torch.float8_e5m2)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=[0.5, 0.2, 0.8]
        )
        print("cuBLAS", ms, min_ms, max_ms)

    def action2config(self, action):
        actions = []
        print(action)
        for index, act in enumerate(action):
            # Normalize the action from [-1, 1] to [0, 1]
            normalized_action = (act + 1) / 2

            # Map the normalized value to an index in the range
            key = round(normalized_action * (len(self.action_values[index]) - 1))

            # Select the corresponding power of 2 value
            actions.append(self.action_values[index][key])
        print(
            f"BLOCK_SIZE_M= {actions[0]}, BLOCK_SIZE_N= {actions[1]}, BLOCK_SIZE_K= {actions[2]}, GROUP_SIZE_M= 8, num_stages={actions[3]}, num_warps={actions[4]}"
        )

    def get_datasizes(self):
        return self.datasizes

    def close(self): ...
