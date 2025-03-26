import os
import paramiko
import getpass
from scp import SCPClient

# === CONFIGURATION ===
REMOTE_HOST = "appa"  # Change this to your server's address
REMOTE_USER = "reiber"     # Change this to your SSH username

GATEWAY_USER = "reiber"
GATEWAY_HOST = "ssh1.cs.uni-tuebingen.de"

BASE_DIR = "/local/reiber/hannah/experiments/"
EXPERIMENT = "pub_samos25"
MODEL = "embedded_vision_net"
FILE = "history.yml"
# FILE = "train.log"
RUNS = [
    # "ae_nas_cifar10_w50000_nopred/embedded_vision_net",
    # "ae_nas_cifar10_w50000_bounds/embedded_vision_net",
    # "ae_nas_cifar10_w50000_groupedpw/embedded_vision_net_groupedpw",
    # "ae_nas_cifar10_w50000_groupedpw_fullparams/embedded_vision_net_groupedpw",
    # "ae_nas_cifar10_w50000_iso/embedded_vision_net",
    # "ae_nas_cifar10_w50000_bounds_iso/embedded_vision_net",
    # "ae_nas_cifar10_w200000_nopred/embedded_vision_net",
    # "ae_nas_cifar10_w200000_bounds/embedded_vision_net",
    # "ae_nas_cifar10_w500000_nopred/embedded_vision_net",
    # "ae_nas_cifar10_w500000_m128000000/embedded_vision_net",
    # "ae_nas_cifar10_w500000_m150000000/embedded_vision_net",
    # "ae_nas_cifar10_w500000_m128000000_pre/embedded_vision_net",
    # "ae_nas_cifar10_w2000000_nopred/embedded_vision_net",
    # "ae_nas_cifar100_w500000/embedded_vision_net",
    # "ae_nas_cifar100_w500000_bounds/embedded_vision_net",
    # "ae_nas_cifar100_w500000_grouped/embedded_vision_net_groupedpw",
    # "ae_nas_cifar10_unconstrained/embedded_vision_net",
    "ae_nas_cifar10_unconstrained_large/embedded_vision_net_large",
    "ae_nas_cifar10_w4M_m40M/embedded_vision_net"
    # "random_nas_cifar10",
]  # Add more file paths as needed

REMOTE_FILES = [BASE_DIR + EXPERIMENT + "/trained_models/" + r + "/" + FILE for r in RUNS]

LOCAL_BASE_DIR = "."  # Change this to where you want files to be stored

# === GET PASSWORD ONCE ===
gateway_password = getpass.getpass(f"Enter SSH password for {GATEWAY_USER}@{GATEWAY_HOST}: ")
remote_password = getpass.getpass(f"Enter SSH password for {REMOTE_USER}@{REMOTE_HOST}: ")

# === SET UP SSH CONNECTIONS ===
try:
    # Connect to Gateway (Jump Host)
    gateway_ssh = paramiko.SSHClient()
    gateway_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    gateway_ssh.connect(GATEWAY_HOST, username=GATEWAY_USER, password=gateway_password)

    # Create a proxy transport through the gateway
    transport = gateway_ssh.get_transport().open_channel(
        "direct-tcpip", (REMOTE_HOST, 22), ("localhost", 0)
    )

    # Connect to the Remote Server via the Proxy
    remote_ssh = paramiko.SSHClient()
    remote_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    remote_ssh.connect(REMOTE_HOST, username=REMOTE_USER, password=remote_password, sock=transport)

    with SCPClient(remote_ssh.get_transport()) as scp:
        for remote_file, run in zip(REMOTE_FILES, RUNS):
            local_file = os.path.join(LOCAL_BASE_DIR, "trained_models/", run, FILE)
            local_dir = os.path.dirname(local_file)

            # Create necessary folder structure if it doesn't exist
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            # Check if file exists and prompt for overwrite
            if os.path.exists(local_file):
                # choice = input(f"File {local_file} exists. Overwrite? (y/n): ").strip().lower()
                choice = "y"
                if choice != "y":
                    print(f"Skipping {remote_file}")
                    continue

            print(f"Copying {remote_file} to {local_file} ...")
            scp.get(remote_file, local_file)

    print("✅ All files copied successfully!")

except Exception as e:
    print(f"❌ Error: {e}")

finally:
    remote_ssh.close()
    gateway_ssh.close()
