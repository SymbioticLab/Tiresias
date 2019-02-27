from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


'''
JOB status:
ADDED: add job into JOBS
EVENT: init job events into event list
PENDING:
RUNNING: running job
END: completed
ERROR
'''
import numpy
import math
import util
import models
import csv
import time
import sys
# import cluster
# from switch import _Switch
# from node import _Node
# from cluster import _Cluster

# #get host info
# CLUSTER = cluster.CLUSTER
import flags
FLAGS = flags.FLAGS

class _TFJobs(object):

    '''
    nested-class g_job
    '''
    class g_job(object):
        def __init__(self, num_gpu, total_gpu=0):
            self.num_gpu = num_gpu       
            self.name = str(num_gpu) + '-GPU'
            self.total_job = 0
            self.end_job = 0
            self.num_queue = 2
            self.queues = [list() for i in range(self.num_queue)]
            self.queue_limit = [3600, 7200, 18000]
            self.total_gpu = total_gpu
            self.free_gpu = total_gpu
            self.running_jobs = list()
            self.pending_jobs = list()
            self.runnable_jobs = list()

        def alloc_free_gpus(self, need_num):
            if self.free_gpu >= need_num:
                self.free_gpu -= need_num
                return True
            else:
                return False

        def release_job_gpu(self, num_job=1):
            if num_job < 0:
                util.print_fn("Error: num_job < 0")
                exit()
            self.free_gpu += int(self.num_gpu * num_job)

        def empty_gpu_alloc(self):
            self.free_gpu = self.total_gpu

        def get_gpu_reservation(self, reserved_num):
            '''
            Cluster manager should decide (dynamically) the reserved gpus for each g_job object
            '''
            # diff_gpu = reserved_num - self.total_gpu
            # self.total_gpu = reserved_num
            # # how to update free_gpu
            # self.free_gpu += diff_gpu
            used = self.total_gpu - self.free_gpu
            self.total_gpu = reserved_num
            self.free_gpu = self.total_gpu - used


        def get_gpu_demands(self):
            # return int((len(self.running_jobs) + len(self.pending_jobs)) * self.num_gpu)
            return int(len(self.runnable_jobs) * self.num_gpu)

    def __init__(self):
        self.num_job = 0        
        self.job_list = list()
        ''' job events is a list of tuple
            (time, dict)
        dict:
            'start_jobs': [xxx,xxx,xxx]
            'end_jobs': [xxx,xxx,xxx]
        '''
        self.job_events = list()        
        #holding pending jobs, add job_idx
        self.pending_jobs = list() # [{job_dict}, {job_dict}]
        self.runnable_jobs = list() # pending + running
        self.running_jobs = list() # running
        self.completed_jobs = list()

        self.migratable_jobs = list()
        self.num_queue = 3
        self.queues = [list() for i in range(self.num_queue)]
        self.queue_limit = [3250, 7200, 18000]

        # mem info in GB
        self.worker_mem = 5
        self.ps_mem = 6
        self.p_w_mem = 0.1

        #sim-gpu-demands
        self.gpu_job = dict()


        #gittins static delta
        self.gittins_delta = 3250

        self.mean_duration = 800
        self.job_dist_data = None

    def get_job_model(self, job_dict):
        # if job_dict.has_key('model_name') and job_dict.has_key('model_scale'):
        if ('model_name' in job_dict) and ('model_scale' in job_dict):
            job_dict['model'] = models.get_model_with_scale(job_dict['model_name'], job_dict['model_scale'])
        else:
            util.print_fn('Not enough model information to get the details')


    def get_network_load(self, job_dict):
        if 'num_gpu' not in job_dict:
            util.print_fn('No gpu information')
            return 

        if 'model' not in job_dict:
            util.print_fn('No model information')
            return
        
        num_w = job_dict['num_gpu']
        num_ps = num_w


        if num_w == 1:
            job_dict['ps_network'] = list()
            job_dict['w_network'] = list([0])

            '''
            check job ps_size 
            '''
            job_dict['ps_ave'] = 0
            return

        job_dict['w_network'] = list([job_dict['model']['total_size']] * num_w)
        job_dict['ps_network'] = list([0] * num_ps)
        for i in range(0, len(job_dict['model']['tensors'])):
            ps_idx = int(i % num_ps)
            # job_dict['ps_network'][ps_idx] += (job_dict['model']['tensors'][i] * num_w)
            job_dict['ps_network'][ps_idx] += (job_dict['model']['tensors'][i])

        for i in range(0, len(job_dict['ps_network'])):
            job_dict['ps_network'][i] = round(job_dict['ps_network'][i], 1)

        '''
        check the PS job size information  
        job_dict['ps_ave'] = round(numpy.mean(job_dict['ps_network']), 1)
        if job_dict['ps_ave'] == 0:
            print(job_dict)

        s_ps_list = sorted(job_dict['ps_network'], reverse=True) 
        job_dict['ps_max'] = s_ps_list[0] 
        max99_idx = int(math.ceil(num_ps * 0.01))
        job_dict['ps_max99th'] = s_ps_list[max99_idx]
        job_dict['ps_max_ave'] = round(job_dict['ps_max'] / job_dict['ps_ave'], 1)
        job_dict['ps_max99_ave'] = round(job_dict['ps_max99th'] / job_dict['ps_ave'], 1)
        '''


    def add_job(self, job_dict):
        ''' Add job (job_dict) into job_list'''
        for key, value in job_dict.items():
        # for key, value in job_dict.iteritems():
            if value is None:
                continue
            if value.isdigit():
                job_dict[key] = int(value)
        job_dict['duration'] = int(float(job_dict['duration']))
        # job_dict['duration'] = int(job_dict['duration'])

        job_dict['rank'] = sys.maxint


        if 'start_time' not in job_dict:
            job_dict['start_time'] = 0
        if 'end_time' not in job_dict:
            job_dict['end_time'] = 0
        if 'pending_time' not in job_dict:
            job_dict['pending_time'] = 0

        if 'submit_time' in job_dict:
            job_dict['r_submit_time'] = int(-1 * job_dict['submit_time'])

        job_dict['start_time'] = sys.maxint
        job_dict['end_time'] = 0
        job_dict['pending_time'] = 0

        # How much time this job has been executed? For preemption algorithms, this should be accumulated
        job_dict['execution_time'] = 0
        job_dict['last_start_time'] = 0
        job_dict['last_check_time'] = 0
        job_dict['executed_time'] = 0

        job_dict['preempt'] = 0
        job_dict['resume'] = 0
        job_dict['promote'] = 0

        job_dict['status'] = 'ADDED'
        job_dict['job_idx'] = len(self.job_list)
        # job_dict['ps'] = int(job_dict['ps'])
        # job_dict['worker'] = int(job_dict['worker'])
        # job_dict['batch_size'] = int(job_dict['batch_size'])
        # job_dict['num_batch'] = int(job_dict['num_batch'])
        # job_dict['sleep'] = int(job_dict['sleep'])

        job_dict['gpus'] = list()
        job_dict['placements'] = list() #prepare an empty job_placement 
        job_dict['ps_placements'] = list()
        job_dict['w_placements'] = list()
        '''
        MS_YARN: only one switch is allowed
        template:
        [{'switch': xx, 'nodes': [{'id':xx, 'num_gpu':xxx}]},
         {'switch': xx, 'nodes': [{'id':xx, 'num_gpu':xxx}, {'id':xx, 'num_gpu':xxx}]},
         {'switch': xx, 'nodes': [{'id':xx, 'num_gpu':xxx}]},
        ]
        '''

        # if ('end_time' in job_dict) and ('duration' not in job_dict):
        #     job_dict['duration'] = job_dict['end_time'] - job_dict['start_time']
        # else:
        #     job_dict['end_time'] = job_dict['start_time'] + job_dict['duration']
        
        if 'model_scale' not in job_dict:
            job_dict['model_scale'] = 1
        #get detailed model inforamtion
        self.get_job_model(job_dict)

        #add job ps/worker information
        self.get_network_load(job_dict)

        self.job_list.append(job_dict)
        self.num_job += 1

        if FLAGS.schedule == 'multi-dlas-gpu':
            num_gpu = job_dict['num_gpu']
            if num_gpu not in self.gpu_job:
                # add that job class
                self.gpu_job[num_gpu] = self.g_job(num_gpu)

            self.gpu_job[num_gpu].total_job += 1


    def print_all_job_size_info(self):
        '''        
        print job tensor info
        '''

        ps_max_ave_fd = open('ps_max_ave.csv', 'w+')
        ps_max_ave_writer = csv.writer(ps_max_ave_fd)  
        ps_max_ave_writer.writerow(['ps_max_ave'])

        ps_max99_ave_fd = open('ps_max99_ave.csv', 'w+')
        ps_max99_ave_writer = csv.writer(ps_max99_ave_fd)  
        ps_max99_ave_writer.writerow(['ps_max99_ave'])

        w_fd = open('w.csv', 'w+')
        w_writer = csv.writer(w_fd)  
        w_writer.writerow(['w'])

        ps_fd = open('ps.csv', 'w+')
        ps_writer = csv.writer(ps_fd)  
        ps_writer.writerow(['ps'])

        ps_w_fd = open('ps_w.csv', 'w+')
        ps_w_writer = csv.writer(ps_w_fd)  
        ps_w_writer.writerow(['ps_w'])

        util.print_fn("Start to dump job information")
        for job in self.job_list:
            if job['ps_ave'] != 0:
                ps_max_ave_writer.writerow(list([job['ps_max_ave']]))
                ps_max99_ave_writer.writerow(list([job['ps_max99_ave']]))
                w_writer.writerow(list([job['w_network'][0]]))
                # ps_w_writer.writerow(job['w_network'][0])
                # for ps in job['ps_network']:
                #     ps_writer.writerow(ps)
                #     ps_w_writer.writerow(ps)
                
                

                
        ps_max_ave_fd.close()
        ps_max99_ave_fd.close()
        w_fd.close()
        ps_fd.close()
        ps_w_fd.close()
        



    def read_job_info(self, job_idx, field=None):
        ''' Read  job information, if field == NONE, show all job info'''
        ''' job_id,num_gpu,submit_time,start_time,duration,model_size,aggr_interval '''
        print('  Job[%d]: ' % job_idx)

        for job in self.job_list:
            if job['job_idx'] == job_idx:
                #find the job
                if field:
                    if isinstance(job[field], int):
                        print('%s :  %d' % (field, job[field]))
                    else:
                        print('%s :  %s' % (field, job[field]))
                else:
                    print(job)
                print('')

    def read_all_jobs(self, field=None):
        for j in self.job_list:
            print('  Job[%d]: ' % j['job_idx'])
            if field:
                if isinstance(j[field], int):
                    print('%s :  %d' % (field, j[field]))
                else:
                    print('%s :  %s' % (field, j[field]))
            else:
                print(j)
            print('')

    def sort_all_jobs(self, mode=None):
        '''
        Sort jobs based on their sumbit_time
        j1, num_gpu, start_t, end_t, duration
        '''
        # tmp_list = sorted(self.job_list, key = lambda e:e.__getitem__('start_time'))
        # tmp_dict = util.search_dict_list(self.job_list, 'start_time', 4)
        # tmp_dict['end_time'] = 15
        # print(tmp_dict)
        # self.job_list = tmp_list

        self.job_list.sort(key = lambda e:e.__getitem__('submit_time'))
        util.print_fn('   Jobs are sorted with their start time')
        # self.read_all_jobs()
        if FLAGS.schedule == 'multi-dlas-gpu' and FLAGS.scheme == 'count':
            for num_gpu, gjob in self.gpu_job.items():
                util.print_fn('%d-GPU jobs have %d ' % (num_gpu, gjob.total_job))

    def create_multi_nodes_placement(self, job, switch_id, node_list):
        tmp_dict = dict() 
        tmp_dict['switch'] = switch_id
        tmp_dict['nodes'] = node_list
        job['placements'].append(tmp_dict)



    def create_single_node_placement(self, job, switch_id, node_id, num_gpu, num_cpu, mem=0):
        '''
        under this switch, there is only one need used
        {'switch': xx, 'nodes': [{'id':xx, 'num_gpu':xxx, 'num_cpu': xxx, 'network': xxxx, 'tasks': [w0, w1, ps1]}]}
        '''
        tmp_dict = dict() 
        tmp_dict['switch'] = switch_id
        node_dict = dict()
        node_dict['id'] = node_id
        node_dict['num_gpu'] = num_gpu
        node_dict['num_cpu'] = num_cpu
        node_dict['mem'] = mem
        node_dict['tasks'] = list()
        # node_dict['network'] = round(sum(job['w_network']) + sum(job['ps_network']), 1)
        node_dict['network'] = 0 #single machine, no network traffic

        tmp_dict['nodes'] = list()
        tmp_dict['nodes'].append(node_dict)
        job['placements'].append(tmp_dict)

        return node_dict['network']

    def remove_from_pending(self, job, event_time):
        job['status'] = 'RUNNING'
        job['start_time'] = event_time
        job['end_time'] = job['start_time'] + job['duration']
        job['pending_time'] = job['start_time'] - job['submit_time']

        self.pending_jobs.remove(job)

    def move_to_pending(self, job):
        job['status'] = 'PENDING'
        self.pending_jobs.append(job)


    def update_pending_time(self, event_time):
        for job in self.pending_jobs:
            if 'sumbit_time' in job:
                job['pending_time'] = int(event_time - job['submit_time'])

    def add_to_runnable(self, job):
        job['status'] = 'PENDING'
        self.runnable_jobs.append(job)

    def push_job_to_running(self, job, event_time):
        if job['status'] != 'PENDING':
            return
        job['status'] = 'RUNNING'
        if job['start_time'] == 0:
            job['start_time'] = event_time
        job['last_start_time'] = event_time


    def sort_shortest_runnable_jobs(self, event_time):
        for job in self.runnable_jobs:
            if job['status'] == 'RUNNING':
                new_execution_time = int(event_time - job['last_check_time'])
                job['execution_time'] = int(job['execution_time'] + new_execution_time)
                job['remaining_time'] = int(job['duration'] - job['execution_time'])

            elif job['status'] == 'PENDING':
                job['execution_time'] = 0
                job['remaining_time'] = int(job['duration'])

            job['last_check_time'] = int(event_time)

        JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_time'))

    def move_to_runnable(self, job):
        ''' job gets into the system: pending or running, and finally END'''
        #job not started yet
        job['status'] = 'PENDING'
        job['start_time'] = sys.maxint
        job['last_start_time'] = 0
        job['last_check_time'] = job['submit_time']
        job['total_executed_time'] = 0 # total
        job['executed_time'] = 0 # used for deciding priority queue, may be zeroed by last_pending_time
        job['pending_time'] = 0
        job['last_pending_time'] = 0 # how much pending_time the job has since last entering the highest priority queue

        if FLAGS.schedule == 'multi-dlas-gpu':
            num_gpu = job['num_gpu']
            self.gpu_job[num_gpu].runnable_jobs.append(job)
        else:
            self.runnable_jobs.append(job)
    
    def update_priority_queues(self, gputime=False):
        for queue in self.queues:
            del queue[:]
        for job in self.runnable_jobs:
            if gputime:
                j_gt = int(job['executed_time'] * job['num_gpu'])
            else:
                j_gt = int(job['executed_time'])

            if j_gt < self.queue_limit[0]:
                self.queues[0].append(job)
                job['q_id'] = 0
            else:
                self.queues[1].append(job)
                job['q_id'] = 1

            # elif j_gt < self.queue_limit[1]:
            #     self.queues[1].append(job)
            #     job['q_id'] = 1
            # elif j_gt < self.queue_limit[2]:
            #     self.queues[2].append(job)
            #     job['q_id'] = 2
            # else:
            #     self.queues[3].append(job)
            #     job['q_id'] = 3

   
    def print_job_events(self):
        util.print_fn('    Print all job events ')
        for event in self.job_events:
            util.print_fn('      event.time[%d], with %d start_jobs, and %d end_jobs' % 
                            (event['time'], len(event['start_jobs']), len(event['end_jobs'])))

        util.print_fn(' ')

    def add_job_end_event(self, job):
        #for job end 
        tmp_dict = util.search_dict_list(self.job_events, 'time', job['end_time'])
        if tmp_dict == None:
            #not found, add the time into to job_events
            tmp_dict = dict()
            tmp_dict['time'] = job['end_time']
            tmp_dict['start_jobs'] = list()
            tmp_dict['end_jobs'] = list()
            tmp_dict['end_jobs'].append(job)
            self.job_events.append(tmp_dict)
        else:
            tmp_dict['end_jobs'].append(job)

        # ''' sort events based on their time'''
        # self.job_events.sort(key = lambda e:e.__getitem__('time'))



    def prepare_job_start_events(self):
        '''
        add job start events into job_events list
        end events should be added when they are starting
        '''
        for job in self.job_list:
            start_t = job['submit_time']
            # util.print_fn('%d, %d' % (start_t, end_t))

            #for job start
            tmp_dict = util.search_dict_list(self.job_events, 'time', start_t)
            if tmp_dict == None:
                #not found, add the time into to job_events
                tmp_dict = dict()
                tmp_dict['time'] = start_t
                tmp_dict['start_jobs'] = list()
                tmp_dict['end_jobs'] = list()
                tmp_dict['start_jobs'].append(job)
                self.job_events.append(tmp_dict)
            else:
                tmp_dict['start_jobs'].append(job)


            job['status'] = 'EVENT' #job has been in EVENT status

        ''' sort events based on their time'''
        self.job_events.sort(key = lambda e:e.__getitem__('time'))
        util.print_fn('Init, add job start events')
        self.print_job_events()


    def add_migratable(self, job):
        '''
        add job into migratable job list 
        1. distributed jobs
        2. running jobs
        3. ?
        '''
        if job['num_w'] <= 1:
            return

        #if job is distributed ?

        if job not in self.migratable_jobs:
            self.migratable_jobs.append(job)            


    def remove_migratable(self, job):
        '''
        remove from migratable job list

        '''
        if job in self.migratable_jobs:
            self.migratable_jobs.remove(job)


    def add_gpu_job(self, job):
        '''
        only used in sim-gpu-demands
        '''
        num_gpu = job['num_gpu']
        if num_gpu not in self.gpu_job:
            self.gpu_job[num_gpu] = 0
        self.gpu_job[num_gpu] = self.gpu_job[num_gpu] + 1

    def delete_gpu_job(self, job):
        num_gpu = job['num_gpu']
        if num_gpu not in self.gpu_job:
            print("Error in release_gpu_job")

        self.gpu_job[num_gpu] = self.gpu_job[num_gpu] - 1

    def end_job(self, e_job):
        if FLAGS.schedule != 'multi-dlas-gpu':
            util.print_fn("Not multi-dlas-gpu")
            exit()
        
        num_gpu = e_job['num_gpu']
        gjob = self.gpu_job[num_gpu]
        gjob.release_job_gpu(1)
        gjob.runnable_jobs.remove(e_job)
        # gjob.running_jobs.remove(e_job)
        gjob.queues[e_job['q_id']].remove(e_job)       
        gjob.end_job += 1


    def init_reserve_gpus(self, total_num):
        num_group = len(self.gpu_job)
        ave_gpu = math.floor(total_num / num_group)
        for num_gpu, gjob in self.gpu_job.items():
            gjob.get_gpu_reservation(ave_gpu)

    def reserve_gpus(self, total_num):
        '''
        GPU cluster reserve gpus for gpu_job groups
        '''
        num_group = len(self.gpu_job)
        ave_gpu = math.floor(total_num / num_group)

        job_list = list()
        for num_gpu, gjob in self.gpu_job.items():
            tmp_dict = dict()
            tmp_dict['num_gpu'] = num_gpu
            tmp_dict['used_gpu'] = gjob.total_gpu - gjob.free_gpu
            tmp_dict['demands'] = gjob.get_gpu_demands()
            tmp_dict['cur_gpu'] = gjob.total_gpu
            tmp_dict['cur_free_gpu'] = gjob.free_gpu
            tmp_dict['reserve'] = 0
            job_list.append(tmp_dict)

        total_free_gpu = total_num - sum(k['used_gpu'] for k in job_list) 
        total_demands = sum(k['demands'] for k in job_list)
        # print('total_free %d, total_demands %d' % (total_free_gpu, total_demands))
        if total_demands == 0: 
            return
        
        '''demand-based, keep current used_gpu'''
        remain_free_gpu = total_free_gpu
        job_list.sort(key = lambda e:e.__getitem__('demands'))
        for job_dict in job_list:
            if job_dict['demands'] == 0:
                continue

            ratio = round((job_dict['demands'] * 1.0) / total_demands, 2)
            cal_gpu = int(math.floor((ratio * total_num) / job_dict['num_gpu']) * job_dict['num_gpu'])
            cal_gpu = job_dict['demands'] if job_dict['demands'] <= cal_gpu else cal_gpu
            extra_gpu = cal_gpu - job_dict['used_gpu']
            if extra_gpu <= 0:
                extra_gpu = 0
            elif extra_gpu > remain_free_gpu:
                extra_gpu = int(math.floor(remain_free_gpu / job_dict['num_gpu']) * job_dict['num_gpu'])

            # print('%d-GPU, u%d, cal_gpu %d, extra_g %d' %(job_dict['num_gpu'], job_dict['used_gpu'], cal_gpu, extra_gpu))
            job_dict['reserve'] = job_dict['used_gpu'] + extra_gpu
            remain_free_gpu -= extra_gpu
            # if remain_free_gpu <= 0:
            #     break

        ''' still remaining, give to the right job group'''
        job_list.sort(key = lambda e:e.__getitem__('num_gpu'))
        num_full = 0
        while remain_free_gpu > 0:
            # if all are satisfied
            if num_full >= len(job_list):
                break
            else:
                num_full = 0

            for job_dict in job_list:
                if job_dict['demands'] <= job_dict['reserve']:
                    num_full += 1
                    continue
                if remain_free_gpu >= job_dict['num_gpu']:                
                    remain_free_gpu -= job_dict['num_gpu']
                    job_dict['reserve'] += job_dict['num_gpu']
                else:
                    num_full += 1

                if remain_free_gpu <= 0: 
                    break

        #execute reservation
        for job_dict in job_list:
            num_gpu = job_dict['num_gpu']
            self.gpu_job[num_gpu].get_gpu_reservation(job_dict['reserve'])
            print("%d-j, T%d, F%d, U%d, N%d, R%d; " % (job_dict['num_gpu'], job_dict['cur_gpu'], job_dict['cur_free_gpu'], job_dict['used_gpu'], job_dict['demands'], job_dict['reserve']), end=' ')

        for num_gpu, gjob in self.gpu_job.items():
            if gjob.free_gpu < 0:
                print("Error free gpu, %d" % num_gpu)
                exit()


        util.print_fn(' %s is done' % sys._getframe().f_code.co_name)

    def completion_check(self):
        for num_gpu, gjob in self.gpu_job.items():
            if gjob.end_job != gjob.total_job:
                util.print_fn('!!!! Miss-match %d completed jobs with %d total jobs in %d-GPU jobs' % (gjob.end_job, gjob.total_job, num_gpu))

    def test_reserve_gpus(self, total_num):
        for num_gpu, gjob in self.gpu_job.items():
            gjob.total_gpu = 0
            gjob.free_gpu = 0
            gjob.runnable_jobs = []

        self.gpu_job[8].total_gpu = 32
        self.gpu_job[8].free_gpu = 0 
        self.gpu_job[8].runnable_jobs.extend([4,5,6,7,8])

        self.gpu_job[16].total_gpu = 32 
        self.gpu_job[16].free_gpu = 16
        self.gpu_job[16].runnable_jobs.extend([5,6,7,8,9])

        self.reserve_gpus(total_num)

JOBS = _TFJobs()


_allowed_symbols = [
    'JOBS'
]
