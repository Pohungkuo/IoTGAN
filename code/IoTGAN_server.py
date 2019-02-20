import sys
task_number=int(sys.argv[1])
import tensorflow as tf


#cluster = tf.train.ClusterSpec({"PS":["localhost:2222"], "DSGraph":["localhost:2223", "localhost:2224"]})
cluster = tf.train.ClusterSpec({"PS":["192.168.0.101:2222"], "DSGraph":["192.168.0.101:2223", "192.168.0.103:2224"]})

server = tf.train.Server(cluster, job_name="PS", task_index=task_number)

print("Server starting #{}".format(task_number))

#server.start()
server.join()
