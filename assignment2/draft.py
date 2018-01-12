import tensorflow as tf
import numpy as np 

n_features = 2

input_placeholder = tf.placeholder(tf.int32,shape = (None, n_features)) 

pretrained_embeddings = np.array([[1,0,0],[0,2,0],[0,0,3],[4,0,0],[0,5,0]])
lookup_result = tf.nn.embedding_lookup(pretrained_embeddings, input_placeholder)
embedding = tf.reshape(tensor = lookup_result, shape = [-1,n_features*3])


sess = tf.Session()
sess.run(tf.initialize_all_variables())

tester = embedding.eval(feed_dict = {input_placeholder:np.array([[0,2],[2,4],[1,1]])},session = sess)

print(tester.shape)
print(tester)



print(pretrained_embeddings.shape[-1])