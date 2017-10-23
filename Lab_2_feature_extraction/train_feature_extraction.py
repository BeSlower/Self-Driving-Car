import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

# TODO: Load traffic signs data.
with open('./train.p', 'rb') as f:
    train = pickle.load(f)

# TODO: Split data into training and validation sets.
data = train['features']
label = train['labels']
num_classes = len(list(set(label)))

x_train, x_valid, y_train, y_valid = train_test_split(data, label, test_size=0.33, random_state=0)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))
y = tf.placeholder(tf.int64, None)
one_hot_y = tf.one_hot(y, num_classes)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.


shape = (fc7.get_shape().as_list()[-1], num_classes)
w_8 = tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=1e-2))
bias_8 = tf.Variable(tf.zeros(num_classes))
logits = tf.nn.xw_plus_b(fc7, w_8, bias_8)

# TODO: Define loss, training, accuracy operations.
rate = 1e-3
epoches = 50
batch_size = 128

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.arg_max(logits, 1), tf.arg_max(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def eval_on_data(X, y, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        loss, acc = sess.run([loss_operation, accuracy_operation], feed_dict={x: X_batch, y: y_batch})
        total_loss += (loss * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])

    return total_loss/X.shape[0], total_acc/X.shape[0]

# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Start training ...')
    for epoch in range(epoches):
        x_train, y_train = shuffle(x_train, y_train)

        for offset in range(0, x_train.shape[0], batch_size):
            end = offset + batch_size
            batch_x = x_train[offset:end]
            batch_y = y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        print(x_valid.shape)
        print(y_valid.shape)
        valid_loss, valid_acc = eval_on_data(x_valid, y_valid, sess)
        print("Epoch {}: valid loss {:.4f}, valid accuracy {:.2f}".format(epoch+1, valid_loss, valid_acc))