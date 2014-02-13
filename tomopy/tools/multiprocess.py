# -*- coding: utf-8 -*-
import multiprocessing as mp


class multiprocess(object):
    def __init__(self, target_func, num_processes=mp.cpu_count(), **kwargs):
        self.total_jobs = 0
        self.jobs = mp.JoinableQueue()
        self.results = mp.Queue()

        tup = (self.jobs, self.results)
        self.num_processes = num_processes
        self.p = [mp.Process(target=target_func, 
                             args=tup) for i in range(num_processes)]

        for process in self.p:
            process.start()

    def add_job(self, job):
        self.total_jobs += 1
        self.jobs.put(job)

    def close_out(self):
        # Add Poison Pills
        for i in range(self.num_processes):
            self.jobs.put((None,))

        completed_jobs = 0
        res_list = []
        while True:
            if not self.results.empty():
                res_list.append(self.results.get())
                completed_jobs += 1
            if completed_jobs==self.total_jobs:
                break
        
        self.jobs.join()
        self.jobs.close()
        self.results.close()

        for process in self.p:
            process.join()
            
        return res_list


def worker(func):
    def worker_in(*args, **kwargs):
	#name = mp.current_process().name
        jobs_completed = 0
        jobs, results = args
        while True:
            job_args = jobs.get()
            if job_args[0] is None: # Deal with Poison Pill
                #print '{}: Exiting. {:d} jobs completed.'.format(name, jobs_completed)
                jobs.task_done()
                break
            res = func(job_args)
            jobs_completed += 1
            jobs.task_done()
            results.put(res)
        return worker_in
    return worker_in