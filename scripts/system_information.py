# Imports
import psutil as ps
import GPUtil as gpx
import platform
from tabulate import tabulate
from datetime import datetime

# Simple helper function to scale to higher values
def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


# First let's define a function to display the system's information
def print_system_info():
    # System Info
    print("="*20, "System Information", "="*20)
    uname = platform.uname()
    print(f"System: {uname.system}")
    print(f"Release: {uname.release}")
    print(f"Processor: {uname.machine}")

    print("\n")

    # CPU Info
    # let's print CPU information
    print("="*25, "CPU Info", "="*25)
    # number of cores
    print("Physical cores:", ps.cpu_count(logical=False))
    print("Total cores:", ps.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = ps.cpu_freq()
    print(f"Current Frequency: {cpufreq.current:.2f}Mhz")

    print("\n")

    # Memory Information
    print("="*20, "Memory Information", "="*20)
    # get the memory details
    svmem = ps.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")
    print(f"Available: {get_size(svmem.available)}")
    print(f"Used: {get_size(svmem.used)}")
    print(f"Percentage: {svmem.percent}%")

    print("\n")

    # Swap Information
    print("="*27, "SWAP", "="*27)
    swap = ps.swap_memory()
    print(f"Total: {get_size(swap.total)}")
    print(f"Free: {get_size(swap.free)}")
    print(f"Used: {get_size(swap.used)}")
    print(f"Percentage: {swap.percent}%")

    print("\n")

    # Disk Information
    print("="*21, "Disk Information", "="*21)
    print("Partitions and Usage:")
    partitions = ps.disk_partitions()
    # I want only to get the information about the main SSD where Windows is installed
    print(f"=== Device: {partitions[0].device} ===")
    print(f"  Mountpoint: {partitions[0].mountpoint}")
    print(f"  File system type: {partitions[0].fstype}")
    partition_usage = ps.disk_usage(partitions[0].mountpoint)
    print(f"  Total Size: {get_size(partition_usage.total)}")
    print(f"  Used: {get_size(partition_usage.used)}")
    print(f"  Free: {get_size(partition_usage.free)}")
    print(f"  Percentage: {partition_usage.percent}%")

    print("\n")

    # GPU Information
    print("="*24, "GPU Details", "="*24)
    gpus = gpx.getGPUs()
    list_gpus = []
    for gpu in gpus:
        # name of GPU
        gpu_name = gpu.name
        # get % percentage of GPU usage of that GPU
        gpu_load = f"{gpu.load*100}%"

        gpu_free_memory = f"{round(gpu.memoryFree/1024,2)}GB"
        gpu_used_memory = f"{round(gpu.memoryUsed/1024,2)}GB"
        gpu_total_memory = f"{round(gpu.memoryTotal/1024,2)}GB"
        gpu_temperature = f"{gpu.temperature} °C"
        list_gpus.append((
            gpu_name, gpu_free_memory, gpu_used_memory,
            gpu_total_memory, gpu_temperature
        ))
    print(tabulate(list_gpus, headers=("Name", "Free VRAM", "Used VRAM", "Total VRAM", "Temperature")))
    print("\n")