import datetime
import time
import datatime

start_dt = datetime.datetime.now()

time.sleep(2)

end_dt = datetime.datetime.now()

cost = end_dt-start_dt

print(cost.seconds)
print(cost.microseconds)
