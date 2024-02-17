import subprocess

download_command = "wget -q https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.4.1/auto_gptq-0.4.1+cu118-cp310-cp310-linux_x86_64.whl"
subprocess.run(download_command, shell=True)

install_command = "pip install -qqq auto_gptq-0.4.1+cu118-cp310-cp310-linux_x86_64.whl --progress-bar off"
subprocess.run(install_command, shell=True)

print("Download complete")
