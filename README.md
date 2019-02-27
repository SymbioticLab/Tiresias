Tiresias -- A GPU Cluster Manager for Distributed Deep Learning Training without complete job information
====

Tiresias is a GPU cluster resource manager that aims at minimizing distributed deep learning (DDL) jobs’ completion times with partial or no a priori knowledge. It does not rely on any intermediate DL algorithm states (e.g., training loss values) or framework specifics (e.g., tensors-to-parameter server mapping).

DDL training jobs bring some unique challenges to the cluster manager:
1. unpredictable training time 
2. over-aggressive job consolidation 
3. all-or-nothing resource allocation
4. inflexibility in GPU sharing (job preemption and resumption)

Tiresias tackles those challenges with the **Discretized-2DAS** (two-dimensional age/attained-service based) scheduler and the model profile-based job placement scheme.
The *2DAS* scheduler, which considers both the spatial (GPU requirements) and temporal (job's executed time) aspects of DDL jobs, has two scheduling algorithms (*Discretized 2D-LAS* and *Discretized 2D-Gittins Index*). They can minimize the average JCT with no and partial job knowledge, respectively. 
The profile-based job placement scheme can appropriately relax the consolidation constraints and maintain the resource (GPU) utilization of cluster without hurting jobs’ performance.

Out testbed experiments and large-scale trace-driven simulations show 
that Tiresias improves the average JCT by up to 5.5x (2x) over current production solutions (state-of-the-art DDL cluster scheduler), 
and it performs comparably to the solution using perfect knowledge of all job characteristics.

Detailed design and performance are available in our [NSDI'19 paper](https://www.usenix.org/conference/nsdi19/presentation/gu).


What's in this repository?
-----------

1. Discrete-time simulator of GPU cluster manager for DL training jobs (with both the job scheduler and placement scheme)

**Coming soon ...**  

2. Network(RDMA)-level message profiler for DL models

3. ...

Others
-----------
1. What's **LAS** (Least-Attained Service) algorithm?  
    Nuyens, Misja, and Adam Wierman. "The foreground–background queue: a survey." Performance evaluation 65.3-4 (2008): 286-307.

2. What's **Gittins Index** policy?  
    Gittins, John, Kevin Glazebrook, and Richard Weber. Multi-armed bandit allocation indices. John Wiley & Sons, 2011.