import subprocess
from subprocess import DEVNULL
import psutil
import GPUtil
import shlex
import datetime

class Scheduler():

    class Config():
        def __init__(self, model_name, experiment_id, gsettings, allowed_gpus):
            self.available_cpus = [x for x in range(1, psutil.cpu_count(logical=True))]
            self.allowed_gpus = allowed_gpus
            self.gsettings = gsettings
            self.gpu_memory_load_threshold_pct = float(gsettings["gpu_memory_load_threshold_pct"])
            self.main_memory_load_threshold_pct = float(gsettings["main_memory_load_threshold_pct"])
            self.max_count_running_jobs_per_gpu = int(gsettings["max_count_running_jobs_per_gpu"])
            self.model_name = model_name
            self.experiment_id = experiment_id

    def __init__(self, config):
        self.jobs_finished = 0
        self.task_list = []
        self.jobs_queue = []
        self.config = config

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

    def get_gpu_core_usage_pct(self, gpu_no):
        GPUs = GPUtil.getGPUs()
        for GPU in GPUs:
            if(GPU.id == gpu_no):
                return GPU.load
        return 1


    def get_cpu_usage(self, cpu_no):
        cpu_usages = psutil.cpu_percent(interval=None, percpu=True)
        return cpu_usages[cpu_no] / 100

    def get_main_memory_usage_pct(self):
        return psutil.virtual_memory().percent / 100

    def _add_job_to_gpu(self, variant, gpu_no): 
        cmd = ""
        cmd += self.config.gsettings["python"]
        cmd += " "
        cmd += "-m"
        cmd += " "
        cmd += self.config.gsettings["module"]
        cmd += " "
        cmd += "--model"
        cmd += " "
        cmd += self.config.model_name
        cmd += " "
        cmd += "--experiment-id"
        cmd += " "
        cmd += self.config.experiment_id
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


        args = shlex.split(cmd)

        process = subprocess.Popen(args,stdout=DEVNULL,stderr=DEVNULL)
        self.task_list.insert(0, (gpu_no, process))

    def add_job_to_queue(self, variant):
        self.jobs_queue.insert(0, variant)

    def get_resource_status(self):
        strings_to_print = []
        datetime_str = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        strings_to_print += [f"Time: {datetime_str}"]
        strings_to_print += ["======= Resource status ======="]
        main_memory_usage_pct_str = "{0:.2f}%".format(self.get_main_memory_usage_pct() * 100)
        strings_to_print += [f"Main Memory Load: {main_memory_usage_pct_str}"]
        avg_allowed_cpu = 0
        for cpu_no in self.config.available_cpus:
            avg_allowed_cpu += self.get_cpu_usage(cpu_no)
        avg_allowed_cpu /= len(self.config.available_cpus)
        avg_allowed_cpu_pct_str = "{0:.2f}%".format(avg_allowed_cpu * 100)
        strings_to_print += [f"Average usage of allowed CPUs: {avg_allowed_cpu_pct_str}"]

        for gpu_core_pct, gpu_mem_pct, gpu_no in [(self.get_gpu_core_usage_pct(gpu_no), self.get_gpu_memory_usage_pct(gpu_no), gpu_no) for gpu_no in self.config.allowed_gpus]:
            gpu_core_pct_str = "{0:.2f}%".format(gpu_core_pct * 100)
            gpu_mem_pct_str = "{0:.2f}%".format(gpu_mem_pct * 100)
            strings_to_print += [f"GPU No.{gpu_no}:\tcore={gpu_core_pct_str}\tmem={gpu_mem_pct_str}"]

        return strings_to_print

    def get_job_status(self):
        strings_to_print = ["\n"]
        strings_to_print += ["======= Job status ======="]
        strings_to_print += [f"Jobs: remaining: {len(self.jobs_queue)}\tin-progress: {len(self.task_list)}\tfinished: {self.jobs_finished}"]
        return strings_to_print

    def get_status(self):
        return self.get_resource_status() + self.get_job_status()


    def filtermethod_process(self, item):
        _, process = item
        if(process.poll() == None):
            return True
        else:
            self.jobs_finished += 1
            return False

    def schedule(self):
        strings_to_print = self.get_status()
        self.task_list = [x for x in filter(self.filtermethod_process, self.task_list)]
        if(len(self.jobs_queue) > 0 and self.get_main_memory_usage_pct() < self.config.main_memory_load_threshold_pct):
            gpus_with_usage = sorted([(self.get_gpu_core_usage_pct(gpu_no), self.get_gpu_memory_usage_pct(gpu_no), gpu_no) for gpu_no in self.config.allowed_gpus])
            if(self.get_main_memory_usage_pct() < self.config.main_memory_load_threshold_pct):
                for gpu_core_pct, gpu_mem_pct, gpu_no in gpus_with_usage:
                    if(gpu_mem_pct < self.config.gpu_memory_load_threshold_pct and self.get_count_processes_by_gpu(gpu_no) < self.config.max_count_running_jobs_per_gpu):
                        self._add_job_to_gpu(self.jobs_queue.pop(), gpu_no)
                        strings_to_print += [f"--> Added job to gpu No.{gpu_no}"]
                        break
        for string in strings_to_print:
            print(string)
        print()
        print()

    def has_finished(self):
        return (len(self.task_list) == 0 and len(self.jobs_queue) == 0)
