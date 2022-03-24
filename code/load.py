"""Define functions for data loading and parsing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


class data_loader():
    def __init__(self, batch_size, seq_size, n_channels, n_classes):
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.n_channels = n_channels
        self.n_classes = n_classes

    def load(self, filename, n_samples, n_epochs, shuffle):
        """Read from the given file and construct a dataset for it.

        Args:
            filename: string. Data path to the input file.
            n_samples: int. The number of samples in the input dataset (used for estimating the cache buffer size).
            n_epochs: int. The number of epochs to run. Infinite epochs if set as -1 or None.
            shuffle: boolean. Shuffle the dataset if true.

        Returns:
            dataset: tf.data. A prepared dataset which contains (seqs, labels) tuples:
                seqs: Genome sequences. 3D tensor of [batch_size, length, #nucleotides] size.
                labels: Labels. 2D tensor of [batch_size, number_of_cell_types] size.
        """
        if not tf.io.gfile.exists(filename):
            raise ValueError("Failed to find file: " + filename)

        dataset = tf.data.TextLineDataset(filename)
        dataset = dataset.map(self._parser)
        if shuffle:
            base_fraction_of_examples = 0.1
            base_examples = int(min(n_samples * base_fraction_of_examples, 1000))
            buffer_size = base_examples + 3 * self.batch_size
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(n_epochs)
        return dataset

    ###########
    # private functions
    ###########

    def _parser(self, record):
        """A parser function that splits the given record into a sequence and its labels.

        The input record needs to be a string in the format of "one-hot-sequence;label".
        """
        misc_elements = tf.strings.split(input=record, sep=";")
        seq = tf.strings.to_number(
                tf.reshape(
                    tf.strings.split(input=misc_elements[0], sep=","),
                    [self.seq_size, self.n_channels]), tf.float32)
        label = tf.strings.to_number(
                tf.strings.split(input=misc_elements[1], sep=","), tf.float32)
        label.set_shape([self.n_classes])
        return (seq, label)

#############
# unit test
if __name__ == "__main__":
    loader = data_loader(batch_size=5, seq_size=1000, n_channels=4, n_classes=919)
    data_path = "/storage/pandaman/project/Alzheimers_ResNet/storage/pretrain/seqs_one_hot/validation/data.txt"
    n_rounds = 3
    for (seqs, labels) in loader.load(filename=data_path, n_samples=1000, n_epochs=None, shuffle=False):
        print (seqs)
        print (labels)
        n_rounds -= 1
        if n_rounds <= 0:
            break
########### END OF FILE ###############
