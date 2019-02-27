GPU cluster simulator in Tiresias
===
1. What's included in this folder ?
    1. ``run_sim.py``: The main function script of this simulator
    2. ``flags.py``: Input parser ``class _FlagValues``
    3. ``cluster.py``: GPU cluster ``class _Cluster``; there are multiple switches under the cluster. The infrastructure hierarchy looks like `cluster->switch->node`.
    4. ``switch.py``: Network switch ``class _Switch`` under the cluster. Each switch includes all the server nodes connected under it. From the resource perspective, a switch means a group of nodes. 
    5. ``node.py``: GPU server node ``class _Node`` with resources including CPU, memory, and GPUs
    5. ``jobs.py``: define ``class _TFJobs`` as the collection of DL jobs, and the subclass ``class g_job`` for individual DL job with GPU
    5. ``models.py``: Model information. Based on the given model, get the memory usage (CPU, GPU) and tensor-size (in MB) distribution of the model (10 CNN models from Tensorflow-benchmark)
    6. ``log.py``: Log function for the simulator
    7. ``util.py``: Utility funcitons
    7. ``*_job.csv``: Job trace file, with the following necessary fields: ``job_id,num_gpu,submit_time,iterations,model_name,duration,interval``
    8. ``nxxgxx.csv`` and ``cluster_spec.csv``:  Cluster spec file, including the fields: ``num_switch,num_node_p_switch,num_gpu_p_node,num_cpu_p_node,mem_p_node``
    9. ``yarn-gputxxxx.csv``: The GPU-time distribution of jobs from YARN. The jobs are sampled for every ``xxxx``seconds (1000, 5000, 10000). This is needed for ``Gittins Index`` policy.


2. Before the exection, what's needed?
    1. Infrastructure details
    Define the hierarchy and resource capacity of the infrastructure in ``cluster_spec.csv``. For example, we have a cluster with 4 racks (switches). Under each rack (switch), there are 32 nodes. And each node has 128 CPU cores, 256 GB memory, and 8 GPUs. Then ``cluster_spec.csv`` will look like this:
        ```csv
        num_switch,num_node_p_switch,num_gpu_p_node,num_cpu_p_node,mem_p_node
        4,32,8,128,256
        ```
    2. Job trace
    The job trace to simulate. For each job, the simulator needs the following information:
       * ``job_id``: for tracking
       * ``num_gpu``: gpu requirement
       * ``submit_time``: when the job is submitted. The simulator is event-based and discrete-time. Therefore, the time value starts from ``0``, and in second-scale.
       * ``iterations``: the number of iterations to training. For the scheduling schemes in Tiresias, they are not relying on this information.
       * ``model_name``: what's the model in that job. This is used to estimate the CPU and GPU memory usage, and tensor size (in MB, only consider the large tensors).
       * ``duration``: how long this job will run. This information is used to generate job completion event by the simulator.
       * ``interval``: job submission interval from this job to the next job
    

3. How to run the simulator?
    A simple example of the execution commend should be: (**``python2.7``** is required)
    ```bash
    python2.7 run_sim.py --cluster_spec=n32g4.csv --print --scheme=yarn --trace_file=480_job.csv --schedule=dlas --log_path=test_1
    ```
    The following options are necessary:
    * ``--cluster_spec``: infrastructure spec file
    * ``--trace_file``: job trace
    * ``--scheme``: **placement scheme**
    * ``--schedule``: **scheduler**

    Optional inputs:
    * ``--print``: print debug information
    * ``--log_path``: the output path of the log (cluster, job). The default will be ``time-stamp`` folder under current path

4. What are the placement and scheduling algorithms provided?
    *Placement*: This simulator can't simulate the performance impact from different placement schemes. The job placement only affect resource allocation and job queuing delay.
    * ``count``: just resource counting
    * ``yarn``: get GPUs from the same server nodes under the same switch
    * ``random``: randomly select GPUs from the entire cluster

    *Scheduling*
    * ``fifo``
    * ``fjf``: Fit-job-first. Jobs are scheduled in FIFO order in a best-effort manner. It allows job jumping when following jobs can fit into the available resource (the current job needs more)
    * ``sjf``: Smallest-job-first, in terms of GPU requirement
    * ``lpjf``: longest pending job first
    * ``shorest``: shorestest remaining time job first
    * ``shorest-gpu``: shortest-remaining-gputime job first
    * ``dlas``: discretized LAS (just time-based)
        In ``jobs.py``,  you need to specify ``num_queue`` and ``queue_limit`` for ``MLFQ`` (also for ``dlas-gpu``, and ``gittins``)
        ```python
        # Example1: there are two queues, and the threshold for Q1 is 3600 seconds
        self.num_queue = 2
        self.queue_limit = [3600]

        # Example2: there are four queues, and the threshold for queues is 3600, 7200, 18000 seconds
        self.num_queue = 2
        self.queue_limit = [3600, 7200, 18000]
        ```
    * ``dlas-gpu``: discretized LAS (gpu-time-based)
    * ``gittins``: discretized Gittins Index (gpu-time-based)


5. What's the output?
    Based on the ``--log_path``, all the output files are in that folder (e.g., ``result-20190210-12-20-37`` including:
    1. ``cluster.csv``: cluster-level resource utilization info at each event point
    2. ``jobs.csv``: the job execution information
    3. ``cpu.csv``, ``gpu.csv``, ``memory.csv``, ``network.csv``: those are the utilization details of each resource unit at event points. However, those logs are not accurate under some combinations of placement and scheduler. When ``count`` is chosen, those files are not generated.

    The output logs are defined in ``log.py``; You can modify that file to adjust the output information.


Others
--------------
The code of this part is not polished yet and may be hard to read. For questions, please contact [Juncheng Gu](http://web.eecs.umich.edu/~jcgu/) (jcgu@umich.edu).