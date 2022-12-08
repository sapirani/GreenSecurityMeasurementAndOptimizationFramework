import time

import psutil

x = psutil.disk_io_counters()
print(x)

time.sleep(32)

y = psutil.disk_io_counters()
print(y)
print("read count", y.read_count - x.read_count)
print("read bytes", (y.read_bytes - x.read_bytes)/1024)



