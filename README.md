Tiresias -- An information-agnostic GPU Cluster Manager for Distributed Deep Learning (coming soon)
====

Tiresias is a GPU cluster resource manager that minimizes distributed deep learning (DDL) jobs’ completion times with partial or no a priori knowledge. 
It does not rely on any intermediate DL algorithm states (e.g., training loss values) or framework specifics (e.g., tensors-to-parameter server mapping). 

DDL training jobs bring some unique challenges to their cluster manager: 
(1) unpredictable training time,
(2) over-aggressive job consolidation, and 
(3) inflexibility in GPU sharing (job preemption).
Tiresias tackles those challenges with the proposed *2DAS* scheduler and profile-based job placement scheme.
The *2DAS* scheduler, which considers both the spatial and temporal aspects of DDL jobs, 
has two scheduling algorithms (*Discretized 2D-LAS* and *Discretized 2D-Gittins Index*). 
They can minimize the average JCT with non and partial prior knowledge, respectively. 
The profile-based job placement scheme can appropriately relax the consolidation constraints and maintain the resource (GPU) utilization of cluster without hurting jobs’ performance.

Out testbed experiments and large-scale trace-driven simulations show 
that Tiresias improves the average JCT by up to 5.5x (2x) over current production solutions (state-of-the-art DDL cluster scheduler), 
and it performs comparably to solutions using perfect knowledge of all job characteristics.

Detailed design and performance are available in our [NSDI'19 paper](https://www.usenix.org/conference/nsdi19/presentation/gu).


What's in this repository?
-----------

1. Simulator of GPU cluster manager (both the job scheduler and placement scheme)

**coming soon ...**