import tensorflow as tf
import sys
from model import Model
from dataset import Dataset
from network import *
from datetime import datetime

def main():
    # Dataset path
    train_list = '/path/to/data/flickr_style/train.txt'
    test_list = '/path/to/data/flickr_style/test.txt'

    # Learning params
    learning_rate = 0.001
    training_iters = 12800 # 10 epochs
    batch_size = 50
    display_step = 20
    test_step = 640 # 0.5 epoch

    # Network params
    n_classes = 20
    keep_rate = 0.5

    # Graph input
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_var = tf.placeholder(tf.float32)

    # Model
    pred = Model.alexnet(x, keep_var)

    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # Evaluation
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Init
    init = tf.initialize_all_variables()

    # Load dataset
    dataset = Dataset(train_list, test_list)

    # Launch the graph
    with tf.Session() as sess:
        print 'Init variable'
        sess.run(init)
        
        # Load pretrained model
        load_with_skip('caffenet.npy', sess, ['fc8']) # Skip weights from fc8
        
        print 'Start training'
        step = 1
        while step < training_iters:
            batch_xs, batch_ys = dataset.next_batch(batch_size, 'train')
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_var: keep_rate})
           
            # Display testing status
            if step%test_step == 0:
                test_acc = 0.
                test_count = 0
                for _ in range(int(dataset.test_size/batch_size)):
                    batch_tx, batch_ty = dataset.next_batch(batch_size, 'test')
                    acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, keep_var: 1.})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print >> sys.stderr, "{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_acc)


            # Display training status
            if step%display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                batch_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                print >> sys.stderr, "{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}".format(datetime.now(), step, batch_loss, acc)
     
            step += 1
        print "Finish!"

if __name__ == '__main__':
    main()

