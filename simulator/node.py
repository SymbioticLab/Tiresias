from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import util

'''
TODO: add cpu and network load support in class _Node
'''
class _Node(object):
    def __init__(self, id, num_gpu=0, num_cpu=0, mem=0):
        self.id = id
        self.num_cpu = num_cpu
        self.free_cpus = num_cpu
        self.num_gpu = num_gpu       
        self.free_gpus = num_gpu
        #network load: can be bw, or the amount of traffic
        # in and out should be the same
        self.network_in = 0
        self.network_out = 0

        self.mem = mem
        self.free_mem = mem

        #node class for gandiva
        self.job_gpu = 0
        self.num_jobs = 0

        util.print_fn('    Node[%d] has %d gpus, %d cpus, %d G memory' % (id, num_gpu, num_cpu, mem))
    
    def init_node(self, num_gpu=0, num_cpu=0, mem=0):
        if num_gpu != 0:
            self.num_gpu = num_gpu
            self.free_gpus = num_gpu
        if num_cpu != 0:
            self.num_cpu = num_cpu
            self.free_cpus = num_cpu
        if mem != 0:
            self.mem = mem
            self.free_mem = mem 

        self.add_gpus(self.num_gpu)        
        self.add_cpus(self.num_gpu)        


    ''' GPU  '''
    def add_gpus(self, num_gpu=0):
        pass

    def check_free_gpus(self):
        return self.free_gpus


    def alloc_gpus(self, num_gpu=0):
        '''
        If enough free gpus, allocate gpus
        Return: True, for success;
                False, for failure
        '''
        if num_gpu > self.free_gpus:
            return False
        else:
            self.free_gpus -= num_gpu
            return True

    def release_gpus(self, num_gpu=0):
        '''
        release using gpus back to free list
        '''
        if self.free_gpus + num_gpu > self.num_gpu:
            self.free_gpus = self.num_gpu
            return False
        else:
            self.free_gpus += num_gpu
            return True


    ''' CPU '''

    def add_cpus(self, num_cpu=0):
        pass

    def check_free_cpus(self):
        return self.free_cpus

    def alloc_cpus(self, num_cpu=0):
        '''
        If enough free cpus, allocate gpus
        Return: True, for success;
                False, for failure
        '''
        if num_cpu > self.free_cpus:
            return False
        else:
            self.free_cpus -= num_cpu
            return True

    def release_cpus(self, num_cpu=0):
        '''
        release using cpus back to free list
        '''
        if self.free_cpus + num_cpu > self.num_cpu:
            self.free_cpus = self.num_cpu
            return False
        else:
            self.free_cpus += num_cpu
            return True 


    '''network'''

    def add_network_load(self, in_load=0, out_load=0):
        self.network_in += in_load
        self.network_out += out_load
        self.network_in = round(self.network_in, 1)
        self.network_out = round(self.network_in, 1)


    def release_network_load(self, in_load=0, out_load=0):
        self.network_in -= in_load
        self.network_out -= out_load
        self.network_in = round(self.network_in, 1)
        self.network_out = round(self.network_in, 1)

    def set_network_load(self, in_load=0, out_load=0):
        self.network_in = in_load
        self.network_out = out_load
        self.network_in = round(self.network_in, 1)
        self.network_out = round(self.network_in, 1)


    def alloc_job_res(self, num_gpu=0, num_cpu=0):
        '''
        alloc job resource
        '''
        gpu = self.alloc_gpus(num_gpu)
        cpu = self.alloc_cpus(num_cpu)

        if cpu == False or gpu == False:
            self.release_gpus(num_gpu)
            self.release_cpus(num_cpu)
            return False

        return True 

    def release_job_res(self, node_dict):
        '''
        input is node_dict from placement
        {'id':xx, 'num_gpu':xxx, 'num_cpu': xxx, 'network': xxxx, 'tasks': [w2, ps2]}
        '''
        self.release_network_load(node_dict['network'], node_dict['network'])
        cpu = self.release_cpus(node_dict['num_cpu'])
        gpu = self.release_gpus(node_dict['num_gpu'])

        self.free_mem = self.free_mem + node_dict['mem']

        return (cpu and gpu)

    def release_job_gpu_cpu(self, num_gpu, num_cpu):
        '''
        input is gpu and cpu
        '''
        cpu = self.release_cpus(num_cpu)
        gpu = self.release_gpus(num_gpu)

        return (cpu and gpu)
