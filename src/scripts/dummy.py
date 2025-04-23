import signal
import time


def handler(signum, frame):
    print(f"Caught signal {signum}, cleaning up...")
    time.sleep(5)
    print("Cleanup done.")
    exit(0)


signal.signal(signal.SIGTERM, handler)

print("Sleeping forever. Try: scancel <job_id>")
while True:
    time.sleep(1)
