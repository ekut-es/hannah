import subprocess
import psutil
import GPUtil
import shlex

class Scheduler():
    available_cpus = range(1, 152)
    #allowed_cpus = range(1, 50)
    allowed_cpus = range(1, 152)
    available_gpus = [0, 1, 2, 3]
    allowed_gpus = []
    gpu_memory_load_threshold_pct = 0.9
    main_memory_load_threshold_pct = 0.5
    max_count_running_jobs_per_gpu = 3
    _gsettings = []
    _model_name = ""
    
    jobs_finished = 0
    
    task_list = []
    jobs_queue = []
    
    def set_general_settings(self, gsettings):
        self._gsettings = gsettings
        
    def set_model_name(self, model_name):
        self._model_name = model_name
        
    def set_allowed_gpus(self, gpus):
        self.allowed_gpus = gpus

    def get_count_processes_by_gpu(self, gpu_no):
        counter = 0
        for gpu_no_list, _ in self.task_list:
            if(gpu_no_list == gpu_no):
                counter += 1
        return counter
    
    def get_gpu_memory_usage_pct(self, gpu_no):
        GPUs = GPUtil.getGPUs()
        for GPU in GPUs:
            if(GPU.id == gpu_no):
                return GPU.memoryUtil
        return 1
#        raise Exception(f"GPU No. {gpu_no} not found")
        
    def get_gpu_core_usage_pct(self, gpu_no):
        GPUs = GPUtil.getGPUs()
        for GPU in GPUs:
            if(GPU.id == gpu_no):
                return GPU.load
        return 1
#        raise Exception(f"GPU No. {gpu_no} not found")
        
    def get_cpu_usage(self, cpu_no):
        cpu_usages = psutil.cpu_percent(interval=None, percpu=True)
        return cpu_usages[cpu_no] / 100

    def get_main_memory_usage_pct(self):
        return psutil.virtual_memory().percent / 100
        
    def _add_job_to_gpu(self, variant, gpu_no): 
        cmd = ""
        cmd += self._gsettings["python"]
        cmd += " "
        cmd += "-m"
        cmd += " "
        cmd += self._gsettings["module"]
        cmd += " "
        cmd += "--model"
        cmd += " "
        cmd += self._model_name
        cmd += " "
        cmd += "--experiment-id"
        cmd += " "
        cmd += self._gsettings["experiment_id"]
        cmd += " "
        for key, setting in variant:
            if(isinstance(setting, list)):
                settingstring = str(setting).translate(str.maketrans("", "", ",[]"))
                cmd += f"--{key} {settingstring} "
            elif(isinstance(setting, bool)):
                if(key == "extract_loudest"):
                    if(setting == False):
                        cmd += "--no_extract_loudest "
                else:
                    if(setting == True):
                        cmd += f"--{key} "
            else:
                cmd += f"--{key} {setting} "
        cmd += f"--gpu_no {gpu_no} "
        #cmd += self._gsettings["pipestring"]
        
        
        args = shlex.split(cmd)
        
        with open("out.log","wb") as stdout, open("err.log","wb") as stderr:
            process = subprocess.Popen(args,stdout=stdout,stderr=stderr)
        self.task_list.insert(0, (gpu_no, process))
        
    def add_job_to_queue(self, variant):
        self.jobs_queue.insert(0, variant)
        
    def print_resource_status(self):
        print("===== Resource status=======")
        print(f"Main Memory Load: {round(self.get_main_memory_usage_pct() * 100)}%")
        avg_allowed_cpu = 0
        for cpu_no in self.allowed_cpus:
            avg_allowed_cpu += self.get_cpu_usage(cpu_no)
        avg_allowed_cpu /= len(self.allowed_cpus)
        print(f"Average usage of allowed CPUs: {round(avg_allowed_cpu*100)}%")
        
        for gpu_core_pct, gpu_mem_pct, gpu_no in [(self.get_gpu_core_usage_pct(gpu_no), self.get_gpu_memory_usage_pct(gpu_no), gpu_no) for gpu_no in self.allowed_gpus]:
            print(f"GPU No.{gpu_no}:\tcore={gpu_core_pct*100}%\tmem={gpu_mem_pct*100}%")
            
    def print_job_status(self):
        print("===== Job status=======")
        print(f"Jobs: remaining: {len(self.jobs_queue)}\tin-progress: {len(self.task_list)}\tfinished: {self.jobs_finished}")
            
    def print_status(self):
        print()
        self.print_resource_status()
        print()
        self.print_job_status()
        print()
        
    def filtermethod_process(self, item):
        _, process = item
        if(process.poll() == None):
            return True
        else:
            self.jobs_finished += 1
            return False
        
    def schedule(self):
        self.task_list = [x for x in filter(self.filtermethod_process, self.task_list)]
        if(len(self.jobs_queue) > 0 and self.get_main_memory_usage_pct() < self.main_memory_load_threshold_pct): 
            gpus_with_usage = sorted([(self.get_gpu_core_usage_pct(gpu_no), self.get_gpu_memory_usage_pct(gpu_no), gpu_no) for gpu_no in self.allowed_gpus])
            if(self.get_main_memory_usage_pct() < self.main_memory_load_threshold_pct):
                for gpu_core_pct, gpu_mem_pct, gpu_no in gpus_with_usage:
                    if(gpu_mem_pct < self.gpu_memory_load_threshold_pct and self.get_count_processes_by_gpu(gpu_no) < self.max_count_running_jobs_per_gpu):
                        self._add_job_to_gpu(self.jobs_queue.pop(), gpu_no)
                        print(f"Added job to gpu No.{gpu_no}")
                        break
                    
    def has_finished(self):
        return (len(self.task_list) == 0 and len(self.jobs_queue) == 0)
