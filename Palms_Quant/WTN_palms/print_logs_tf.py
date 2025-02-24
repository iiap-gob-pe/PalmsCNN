import tensorflow as tf
sess = tf.Session()
new_saver = tf.train.import_meta_graph('../model/depth/logs/07122021001808_logs-train-model.meta')
# list all the tensors in the graph
for tensor in tf.get_default_graph().get_operations():
    print (tensor.name)

with tf.Session() as sess:
    # restore the saved vairable
    new_saver.restore(sess, '../model/depth/logs/07122021001808_logs-train-model')
    # print the loaded variable
    print(sess.run(['model_builder/depth/conv1_1/weight_loss:5']))

'''
#new_saver.restore(sess, tf.train.latest_checkpoint('../model/depth/logs/'))
all_vars = tf.get_collection('losses')
for v in all_vars:
    print(v)
    if(str(v)!='Tensor("truediv:0", shape=(), dtype=float32)'):
        v_ = sess.run(v)
        print(v_)
'''