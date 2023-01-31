from os import kill
from os import getpid
from signal import SIGKILL
import psutil


def kill_process():
    # print(getpid())
    current_pid = getpid()

    # Iterate over all running process
    for proc in psutil.process_iter():
        try:
            # Get process name & pid from process object.
            processName = proc.name()
            processID = proc.pid
            # print(dir(proc))
            print(any('joblib' in x.lower() for x in proc.cmdline()))
            # if "python" in processName.lower() and int(current_pid) != int(processID):
            #     kill(processID, SIGKILL)
            # print(processName , ' ::: ', processID)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
