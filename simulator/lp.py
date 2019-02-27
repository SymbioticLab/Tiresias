
"""
set J;
/* there are multiple components in the new job */

set N;
/* nodes in the cluster */


param g{j in J}, integer, >=0;
param c{j in J}, integer, >=0;
param t{j in J}, >=0;
/* resource requirements (GPU, CPU) and network traffic of each component in the new job */

param fg{n in N}, integer, >=0;
param fc{n in N}, integer, >=0;
/* free GPUs and CPUs on each machine */
param ct{n in N}, >=0;
/* current network traffic on each machine */


var p{j in J, n in N}, binary, >= 0;
/* objective variable: placement of the new job */
/* binary is also a constraint */

var maxt;
/* max network traffic load in the cluster */

minimize z: maxt;

s.t. optcon{n in N}: ct[n] + sum{j in J} (p[j, n] * t[j]) <= maxt; 
/* optimization constraint: min(max{network load}) */

s.t. fgcap{n in N}: sum{j in J} (p[j,n] * g[j]) <= fg[n];
/* free gpu constraint */

s.t. fccap{n in N}: sum{j in J} (p[j,n] * c[j]) <= fc[n];
/* free cpu constraint */

s.t. jgcon{j in J}: sum{n in N} (p[j,n] * g[j]) - g[j] = 0;
s.t. jccon{j in J}: sum{n in N} (p[j,n] * c[j]) - c[j] = 0;
/* job gpu / cpu consistency */

"""

from __future__ import print_function

import sys

'''
import cplex
from cplex.exceptions import CplexError

from cplex.six.moves import range
'''

import util
import flags 
import jobs
import cluster
import switch
import node
FLAGS = flags.FLAGS
CLUSTER = cluster.CLUSTER
JOBS = jobs.JOBS


'''
sample data:
j: [ps0, ps1, ps2, w0, w1, w2]
node: [n0, n1]
#gpu requirement
g = [0,0,0,1,1,1]
#cpu requirement
c = [4,4,4,2,2,2]
#network traffic
t = [40,10,10,20,20,20]

t = [15,5,5,15]
num_ps = 4
num_w = 4
m_size = 40
ps_c = 4
w_c = 2

#free cpus on nodes
fc = [40,40]
#free gpus on nodes
fg = [4,4]
#current network load on nodes
ct = [30,0]
'''


def prepare_job_info(new_job):
    '''
    prepare job input
    '''
    ret_dict = dict()
    num_ps = len(new_job['ps_network'])
    num_w = new_job['num_gpu']
    if num_w != 1:
        assert num_ps == num_w
    # print("     job_id: ", new_job['job_id'], " has %d gpus, %d PS\n" % (num_w, num_ps))

    # tmp_g = list([0] * num_ps + [1] * num_w)
    # tmp_c = list([4] * num_ps + [2] * num_w)
    tmp_n = list()
    for ps in new_job['ps_network']:
        # tmp_n.append(round(ps * num_w, 1))
        tmp_n.append(round(ps, 1))
    # for w in new_job['w_network']:
    #     tmp_n.append(w)

    # tmp_cc = list()
    # for i in range(num_ps):
    #     ps_t = new_job['ps_network'][i]
    #     tmp_cc.append(list([0] * num_ps + [ps_t] * num_w))
    # for i in range(num_w):
    #     tmp_list = new_job['ps_network'] + [0] * num_w
    #     tmp_cc.append(tmp_list)
        # tmp_cc.append(list([1] * num_ps + [0] * num_w))
    '''
    cc
        w0 w1
    ps0 xx xx
    ps1 xx xx
    '''
    # for i in range(num_ps):
    #     tmp_cc.append(list([new_job['ps_network'][i]] * num_w))

    # ret_dict['g'] = tmp_g  
    # ret_dict['c'] = tmp_c  
    ret_dict['t'] = tmp_n  
    # ret_dict['cc'] = tmp_cc
    ret_dict['num_ps'] = num_ps
    ret_dict['num_w'] = num_w
    ret_dict['m_size'] = new_job['w_network'][0]
    ret_dict['ps_c'] = 4
    ret_dict['w_c'] = 2
    '''
    ret_dict['m_size']
    '''

    return ret_dict


def prepare_cluster_info():
    tmp_fg = list()
    tmp_fc = list()
    tmp_ct = list()
    ret_dict = dict()
    for switch in CLUSTER.switch_list:
        for node in switch.node_list:
            tmp_fg.append(node.free_gpus)
            tmp_fc.append(node.free_cpus)
            tmp_ct.append(node.network_out)

    ret_dict['fg'] =  tmp_fg
    ret_dict['fc'] =  tmp_fc
    ret_dict['ct'] =  tmp_ct
    ret_dict['num_n'] = CLUSTER.num_node
    return ret_dict



def parse_lp_solution(new_job, result, job_dict, cluster_dict, var_ind):
    num_ps = job_dict['num_ps']
    num_w = job_dict['num_w']
    m_size = job_dict['m_size']
    ps_c = job_dict['ps_c']
    w_c = job_dict['w_c']

    t = job_dict['t']
    # g = job_dict['g']
    # c = job_dict['c']
    # cc = job_dict['cc']
    # num_c = len(job_dict['t'])
    # assert num_c == (num_ps + num_w)
    num_n = len(cluster_dict['fc']) 
    ct = cluster_dict['ct']

    ps_p = var_ind['ps_p']
    nw = var_ind['nw']
    '''
    add job_placement based on Nodes
    '''
    for n_idx in range(num_n):
        num_gpu = 0
        num_cpu = 0
        network = 0
        add_network_load = 0
        
        tmp_nw = result[nw[n_idx]]
        num_cpu += int(tmp_nw * w_c)
        num_gpu += tmp_nw

        # network = ct[n_idx] + round(tmp_nw * m_size, 1)
        add_network_load = round(tmp_nw * m_size, 1)

        tmp_ps_load = 0
        #get the total allocation on each node
        for ps_idx in range(num_ps):
            var = result[ps_p[ps_idx][n_idx]]
            if var != 0:

                tmp_ps_load += round(t[ps_idx] * (num_w - 2 * tmp_nw), 1)
                num_cpu += ps_c
        
        add_network_load += tmp_ps_load
        add_network_load = round(add_network_load, 1)
        num_gpu = int(num_gpu)
        num_cpu = int(num_cpu)
        # if add_network_load > 0:
        #     print("node[%d] add traffic: %.1f" % (n_idx, add_network_load))
        #     network = round(ct[n_idx] + add_network_load, 1)
        #     print("node[%d] total traffic: %.1f" % (n_idx, network))
                
        if num_cpu != 0:
            switch_id = int(n_idx / CLUSTER.num_node_p_switch)
            node_id = int(n_idx % CLUSTER.num_node_p_switch)

            node = CLUSTER.switch_list[switch_id].node_list[node_id]
            #update node
            node.alloc_job_res(num_gpu, num_cpu)
            node.add_network_load(add_network_load, add_network_load)
            #create placement
            tmp_dict = dict() 
            tmp_dict['switch'] = switch_id
            node_dict = dict()
            node_dict['id'] = node_id
            node_dict['num_gpu'] = num_gpu
            node_dict['num_cpu'] = num_cpu
            node_dict['tasks'] = list()
            node_dict['network'] = add_network_load

            tmp_dict['nodes'] = list()
            tmp_dict['nodes'].append(node_dict)
            new_job['placements'].append(tmp_dict)
            # print(tmp_dict)
    # print(new_job['placements'])


def placement(new_job):
    pass