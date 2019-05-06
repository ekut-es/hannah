from ..train import build_config, dump_config, get_output_dir, get_model

from datetime import timedelta
from string import Template
import os.path
import subprocess
import uuid
import spur
import json

file_dir = os.path.dirname(os.path.realpath(__file__))
top_dir = os.path.abspath(os.path.join(file_dir, "..", ".."))

tcml_config = dict(
    tcml_name = 'speech_recognition',
    tcml_ncpus = '4',
    tcml_partition = 'test',
    tcml_mem = '3G',
    tcml_ngpus = '1',
    tcml_estimated_time = "8:00",
    tcml_user = "gerum",
    tcml_user_mail =  "christoph.gerum@uni-tuebingen.de",
    tcml_output_dir = "trained_models",
    tcml_data_dir = "datasets/speech_commands_v0.02/",
    tcml_skip_data = False,
    tcml_simage_dir = "tcml-cluster",
    tcml_skip_simage = False,
    tcml_wd = "/home/gerum/speech_recognition",
    tcml_skip_code = False,
    tcml_master_node = "tcml-master01.uni-tuebingen.de",
    tcml_job_id = ""
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

    filtered_config = {}
    for key, value in config.items():
        if not key.startswith("tcml"):
            filtered_config[key] = value

    rundir = os.path.join(config["tcml_wd"], "runs")
    result = shell.run(["mkdir", "-p", rundir])
    if result.return_code != 0:
        raise Exception("Could not create rundir: {}".format(rundir))

    job_config_path = os.path.join(rundir, job_config_name)
    job_sbatch_path = os.path.join(rundir, job_sbatch_name)

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
    model_name, config = build_config(extra_config=tcml_config)
    print(config["lr"])
    output_dir = get_output_dir(model_name, config)

    #Estimate runtime
    runtime = estimate_duration(config)
    partition = get_partion_name(runtime)
    
    shell = spur.SshShell(
        hostname=config["tcml_master_node"],
        username=config["tcml_user"],
        private_key_file=os.path.expanduser("~/.ssh/id_rsa")
    )

    if not config["tcml_job_id"]:
        config["tcml_job_id"] = str(uuid.uuid4())


        
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
