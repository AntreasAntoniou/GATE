import subprocess

import gradio as gr
import psutil
from rich import print


def get_system_info():
    # get a snapshot on htop and ps aux
    output_htop = subprocess.run(["top", "-n", "1"], capture_output=True)
    output_ps = subprocess.run(["ps", "aux"], capture_output=True)
    output_md_table = f"## System Info\n\n"
    output_md_table += "### htop\n\n"
    output_md_table += f"```\n{output_htop.stdout.decode()}\n```\n\n"
    output_md_table += "### ps aux\n\n"
    output_md_table += f"```\n{output_ps.stdout.decode()}\n```\n\n"
    # get a snapshot on psutil
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage("/").percent
    output_md_table += "### psutil\n\n"
    output_md_table += f"CPU: {cpu_percent}%\n"
    output_md_table += f"Memory: {memory_percent}%\n"
    output_md_table += f"Disk: {disk_percent}%\n"

    return output_md_table


def kill_user_aantoni2():
    subprocess.run(["pkill", "-U", "aantoni2"])
    return "Killed processes for user aantoni2"


def kill_python_processes():
    subprocess.run(["pkill", "-9", "-f", "python"])
    return "Killed all Python processes"


def kill_fish_processes():
    subprocess.run(["pkill", "-9", "-f", "fish"])
    return "Killed all Fish processes"


with gr.Blocks() as app:
    gr.Button("Show System Info").click(get_system_info, outputs=gr.Textbox())
    gr.Button("Kill User aantoni2").click(
        kill_user_aantoni2, outputs=gr.Textbox()
    )
    gr.Button("Kill Python Processes").click(
        kill_python_processes, outputs=gr.Textbox()
    )
    gr.Button("Kill Fish Processes").click(
        kill_fish_processes, outputs=gr.Textbox()
    )

app.launch(share=True, debug=True)
