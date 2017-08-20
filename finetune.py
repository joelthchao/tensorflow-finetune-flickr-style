from datetime import datetime
import sys
import tensorflow as tf

from model import Model
from dataset import Dataset
from network import *


def main():
    if len(sys.argv) != 4:
        print('Usage: python finetune.py train_file test_file weight_file')
        return

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    weight_file = sys.argv[3]

    # Learning params
    learning_rate = 0.001
    training_iters = 12800 # 10 epochs
    batch_size = 50
    display_step = 20
    test_step = 640 # 0.5 epoch

    # Network params
    n_classes = 20
    keep_rate = 0.3

    # Graph input
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_var = tf.placeholder(tf.float32)

    # Model
    pred = Model.alexnet(x, keep_var)

    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # Evaluation
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Init
    init = tf.initialize_all_variables()

    # Load dataset
    dataset = Dataset(train_file, test_file)

    # Launch the graph
    with tf.Session() as sess:
        print('Init variable')
        sess.run(init)

        print('Load pre-trained model: {}'.format(weight_file))
        load_with_skip(weight_file, sess, ['fc8']) # Skip weights from fc8
        
        print('Start training')
        step = 1
        while step < training_iters:
            batch_xs, batch_ys = dataset.next_batch(batch_size, 'train')
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_var: keep_rate})
           
            # Display testing status
            if step % test_step == 0:
                test_acc = 0.
                test_count = 0
                for _ in range(dataset.test_size // batch_size):
                    batch_tx, batch_ty = dataset.next_batch(batch_size, 'test')
                    acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, keep_var: 1.})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print('{} Iter {}: Testing Accuracy = {:.4f}'.format(datetime.now(), step, test_acc), file=sys.stderr)

            # Display training status
            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                batch_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                print('{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}'.format(datetime.now(), step, batch_loss, acc), file=sys.stderr)
     
            step += 1
        print('Finish!')

if __name__ == '__main__':
    main()
