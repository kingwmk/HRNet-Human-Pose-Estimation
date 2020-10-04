import os
import datetime
start_time = str(datetime.datetime.now())
end_time = str(datetime.datetime.now())
file = open("train_time.txt", "w")
file.write(start_time)
file.write(end_time)
file.close()
