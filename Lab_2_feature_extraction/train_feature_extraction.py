import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import time

with open('./train.p', 'rb') as f:
    train = pickle.load(f)

data = train['features']
label = train['labels']
num_classes = len(list(set(label)))

x_train, x_valid, y_train, y_valid = train_test_split(data, label, test_size=0.33, random_state=0)

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, num_classes)

fc7 = AlexNet(resized, feature_extract=True)
fc7 = tf.stop_gradient(fc7)
shape = (fc7.get_shape().as_list()[-1], num_classes)
w_8 = tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=1e-2))
bias_8 = tf.Variable(tf.zeros(num_classes))
logits = tf.nn.xw_plus_b(fc7, w_8, bias_8)

rate = 1e-3
epoches = 50
batch_size = 128

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation, var_list=[w_8, bias_8])
correct_prediction = tf.equal(tf.arg_max(logits, 1), tf.arg_max(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data, sess):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(loss_operation, feed_dict={x: batch_x, y:batch_y})
        total_loss += (loss * len(batch_x))
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples, total_loss / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Start training ...')
    for epoch in range(epoches):
        x_train, y_train = shuffle(x_train, y_train)
        t0 = time.time()
        for offset in range(0, x_train.shape[0], batch_size):
            end = offset + batch_size
            batch_x = x_train[offset:end]
            batch_y = y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy, validation_loss = evaluate(x_valid, y_valid, sess)
        print("Time: %.3f" % (time.time() - t0))
        print("Epoch {}: valid loss {:.4f}, valid accuracy {:.2f}".format(epoch+1, validation_loss, validation_accuracy))