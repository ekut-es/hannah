from ..train import build_config, dump_config, get_output_dir, get_model
from ..config import ConfigOption

from datetime import timedelta
from string import Template
from collections import ChainMap
import os.path
import subprocess
import uuid
import spur
import json
import shutil

file_dir = os.path.dirname(os.path.realpath(__file__))
top_dir = os.path.abspath(os.path.join(file_dir, "..", ".."))

tcml_config = dict(
    tcml_name           = ConfigOption(category="TCML Options",
                                       desc="Name to identify the job in SLURM scheduler",
                                       default='speech_recognition'),
    tcml_ncpus          = ConfigOption(category="TCML Options",
                                       desc="Number of required CPUs for this job",
                                       default=4),
    tcml_partition      = ConfigOption(category="TCML Options",
                                       desc="SLURM partition to use for this job",
                                       required=True,
                                       choices=["test", "day", "week", "month"],
                                       default='test'),
    tcml_mem            = ConfigOption(category="TCML Options",
                                       desc="Required CPU Memory for this job",
                                       default='3G'),
    tcml_ngpus          = ConfigOption(category="TCML Options",
                                       desc="Required number of GPUs for this job",
                                       default='1'),
    tcml_estimated_time = ConfigOption(category="TCML Options",
                                       desc="Reserved runtime for this job. Job will be killed after specified runtime",
                                       default="8:00"),
    tcml_user           = ConfigOption(category="TCML Options",
                                       desc="Username on TCML cluster",
                                       default="gerum"),
    tcml_user_mail      = ConfigOption(category="TCML Options",
                                       desc="Mail address of user",
                                       default= "christoph.gerum@uni-tuebingen.de"),
    tcml_output_dir     = ConfigOption(category="TCML Options",
                                       desc="Name of output directory for trained models on Cluster",
                                       default="trained_models"),
    tcml_data_dir       = ConfigOption(category="TCML Options",
                                       desc="Directory for dataset on Cluster",
                                       default="datasets/"),
    tcml_skip_data      = ConfigOption(category="TCML Options",
                                       desc="Skip synchronization of dataset to cluster",
                                       default=False),
    tcml_simage_dir     = ConfigOption(category="TCML Options",
                                       desc="Directory containing the singularity image",
                                       default="tcml-cluster"),
    tcml_skip_simage    = ConfigOption(category="TCML Options",
                                       desc="Skip synchronisation of singularity image to cluster",
                                       default=False),
    tcml_wd             = ConfigOption(category="TCML Options",
                                       desc="Working directory on cluster",
                                       default="/home/gerum/speech_recognition"),
    tcml_skip_code      = ConfigOption(category="TCML Options",
                                       desc="Skip synchronisation of code to cluster",
                                       default=False),
    tcml_master_node    = ConfigOption(category="TCML Options",
                                       desc="IP address or URL of TCML Master node",
                                       default="tcml-master01.uni-tuebingen.de"),
    tcml_job_id         = ConfigOption(category="TCML Options",
                                       desc="Unique ID of the job if none is given we will automatically generate one",
                                       default=""),
    tcml_ssh_key        =ConfigOption(category="TCML Options",
                                      default="~/.ssh/id_rsa",
                                      desc="Path to the ssh Keyfile used to log in to the tcml-master node")
)


def estimate_duration(config):
    #model = get_model(config)

    estimated_time = timedelta(hours=16)

    return estimated_time

def get_partion_name(estimated_time : timedelta) -> str:

    if estimated_time < timedelta(minutes = 15):
        return "test"
    elif estimated_time < timedelta(days=1):
        return "day"
    elif estimated_time <= timedelta(days=7):
        return "week"
    elif estimated_time <= timedelta(days=30):
        return "month"
    else:
        raise Exception("Jobs with an estimated runtime over one month are not supported on tcml-cluster")

def build_sbatch(config):
    sbatch_template_file = os.path.join(file_dir, "sbatch_file.template")
    sbatch_template = None
    with open(sbatch_template_file) as f:
        sbatch_template = Template(f.read())

    tcml_config = {}
    for key, value in config.items():
        if key.startswith("tcml"):
            tcml_config[key] = str(value) 
            
    sbatch = sbatch_template.safe_substitute(tcml_config)
    return sbatch

def create_wd(config, shell):
    print("Creating working directory")
    result = shell.run(["mkdir", "-p", config["tcml_wd"]])
    if result.return_code != 0:
        raise Exception("Could not create directory {}".format(config["tcml_wd"]))

    result = shell.run(["mkdir", "-p", os.path.join(config["tcml_wd"], config["tcml_data_dir"])])
    if result.return_code != 0:
        raise Exception("Could not create directory {}".format(config["tcml_wd"]))

    result = shell.run(["mkdir", "-p", os.path.join(config["tcml_wd"], config["tcml_output_dir"])])
    if result.return_code != 0:
        raise Exception("Could not create directory {}".format(config["tcml_wd"]))

    result = shell.run(["mkdir", "-p", os.path.join(config["tcml_wd"], config["tcml_simage_dir"])])
    if result.return_code != 0:
        raise Exception("Could not create directory {}".format(config["tcml_wd"]))

    
    print("   done")
    
def sync_data(config):
    print("Synchronizing data folder")

    cmdline = ["rsync", "-rc", config["data_folder"],
                         config["tcml_user"] + "@" + config["tcml_master_node"] + ":" +os.path.join(config["tcml_wd"], config["tcml_data_dir"])]
    
    p = subprocess.Popen(cmdline)
    return p

def sync_code(config):
    print("Synchronizing code folder")

    cmdline = ["rsync", "-rc"]
    cmdline += ["--exclude", "trained_models",
                "--exclude", ".git",
                "--exclude", "orig",
                "--exclude", ".mypy_cache/",
                "--exclude", "datasets"]

    cmdline +=  [top_dir + "/",
                 config["tcml_user"] + "@" + config["tcml_master_node"] + ":" +config["tcml_wd"] + "/"]
    
    print(" ".join(cmdline))

    
    p = subprocess.Popen(cmdline)
    return p

    
def sync_simage(config):
    print("Synchronizing singularity image")

    cmdline = ["rsync", "-rc",
               os.path.join(top_dir, "tcml-cluster", "Speech-Recognition.simg"),
               config["tcml_user"] + "@" + config["tcml_master_node"] + ":" + os.path.join(config["tcml_wd"], config["tcml_simage_dir"])]
    p = subprocess.Popen(cmdline)
    return p

def enqueue_job(sbatch, config, shell):
    job_id = config["tcml_job_id"]

    job_config_name = "config." + job_id + ".json"
    job_sbatch_name = "job." + job_id + ".sbatch"
    job_compress_name = "compress." + job_id + ".yaml"

    filtered_config = {}
    for key, value in config.items():
        if not key.startswith("tcml"):
            filtered_config[key] = value

    compress_file = config['compress']
    

    rundir = os.path.join(config["tcml_wd"], "runs")
    result = shell.run(["mkdir", "-p", rundir])
    if result.return_code != 0:
        raise Exception("Could not create rundir: {}".format(rundir))

    job_config_path = os.path.join(rundir, job_config_name)
    job_sbatch_path = os.path.join(rundir, job_sbatch_name)
    job_compress_path = os.path.join(rundir, job_compress_name)    
	
    if compress_file:
        with shell.open(job_compress_path, "w") as target:
            with open(compress_file, "r") as source:
                shutil.copyfileobj(source, target)
        config["compress"] = job_compress_path

    with shell.open(job_config_path, "w") as f:
        s = json.dumps(dict(filtered_config), default=lambda x: str(x), indent=4, sort_keys = True)
        f.write(s)

    with shell.open(job_sbatch_path, "w") as f:
        f.write(sbatch)

    print("Starting job:")
    res = shell.run(["sbatch", job_sbatch_path], cwd=config["tcml_wd"])
    if result.return_code != 0:
        raise Exception("Could not start job: {}".format(job_sbatch_path))
    
    
# Run training on tcml machine learning cluster
def main():
    user_config_name = os.path.join(top_dir, "tcml_config.json")
    user_config = {}
    if os.path.exists(user_config_name):
        with open(user_config_name) as config_file:
            user_config = json.load(config_file)
        
    
    model_name, config = build_config(extra_config=ChainMap(user_config, tcml_config))

    if not config["tcml_job_id"]:
        config["tcml_job_id"] = str(uuid.uuid4())

    data_folder_name = os.path.basename(os.path.normpath(config["data_folder"]))
    config["tcml_data_dir"] = os.path.join(config["tcml_data_dir"],
                                           data_folder_name)
    
        
    #Prefix ids with model name
    config["tcml_name"] =  config["model_name"] + "_" + config["tcml_name"]
    config["tcml_job_id"] =  config["model_name"] + "_" + config["tcml_job_id"]

    output_dir = get_output_dir(model_name, config)

    #Estimate runtime
    runtime = estimate_duration(config)
    partition = get_partion_name(runtime)
    
    shell = spur.SshShell(
        hostname=config["tcml_master_node"],
        username=config["tcml_user"],
        private_key_file=os.path.expanduser(config["tcml_ssh_key"])
    )

        
    create_wd(config, shell)

    if not config["tcml_skip_data"]:
        sync_data_job = sync_data(config)

    if not config["tcml_skip_simage"]:
        sync_simage_job = sync_simage(config)

    if not config["tcml_skip_code"]:
        sync_code_job = sync_code(config)

    if not config["tcml_skip_data"]:
        sync_data_job.wait()
    
    if not config["tcml_skip_simage"]:
        sync_simage_job.wait()

    if not config["tcml_skip_code"]:
        sync_code_job.wait()

    print("Finished synchronization jobs")
    sbatch = build_sbatch(config)
    enqueue_job(sbatch, config, shell)
    
    
if __name__ == "__main__":
    main()
