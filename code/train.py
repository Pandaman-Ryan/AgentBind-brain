import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models, initializers, regularizers, callbacks
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from load import data_loader


"""Abstract base model.

This class includes a skeleton of a machine learning model and implements two
fundamental modules, the convolutional layer and the fully connected layer.
"""
class base_model(tf.keras.Model):
    def __init__(self, seq_length, n_channels, n_classes):
        super(base_model, self).__init__()
        self.seq_length = seq_length
        self.n_channels = n_channels
        self.n_classes = n_classes
    
    def build_model(self, input_shape, training=True):
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x, training))

    def call(self, inputs, training=False):
        raise NotImplementedError()

    def _conv_layer(self, filters, kernal_size, layer_id, use_relu):
        conv_layer = layers.Conv1D(
                filters=filters, kernel_size=kernal_size, strides=1, padding='same',
                kernel_initializer=initializers.HeUniform(),
                kernel_regularizer=regularizers.l1_l2(l1=1e-11, l2=1e-11),
                bias_initializer=initializers.Constant(value=0),
                name='conv_%s' %(layer_id))
        bn_layer = layers.BatchNormalization(name = 'bn_%s' %(layer_id))
        if use_relu:
            relu_layer = layers.Activation(activation='relu', name='relu_%s' %(layer_id))
            return conv_layer, bn_layer, relu_layer
        else:
            return conv_layer, bn_layer

    def _fc_layer(self, units, layer_id, post_processing):
        fc_layer = layers.Dense(units=units,
                kernel_initializer=initializers.HeUniform(),
                kernel_regularizer=regularizers.l1_l2(l1=1e-11, l2=1e-11),
                bias_initializer=initializers.Constant(value=0),
                name='fc_%s' %(layer_id))
        if post_processing:
            bn_layer = layers.BatchNormalization(name = 'bn_%s' %(layer_id))
            relu_layer = layers.Activation(activation='relu', name='relu_%s' %(layer_id))
            return fc_layer, bn_layer, relu_layer
        else:
            return fc_layer

"""ResNet component for L1, L2, and L4."""
class resnet_block_uniform_kernels(base_model):
    def __init__(self, seq_length, n_channels, n_classes, filters, kernal_size, block_id):
        super(resnet_block_uniform_kernels, self).__init__(seq_length, n_channels, n_classes)
        self.conv_layer_1, self.bn_layer_1, self.relu_layer_1 = self._conv_layer(
                filters=filters, kernal_size=kernal_size,
                layer_id="%s_layer_%d" %(block_id,1), use_relu=True)
        self.conv_layer_2, self.bn_layer_2 = self._conv_layer(
                filters=filters, kernal_size=kernal_size,
                layer_id="%s_layer_%d" %(block_id,2), use_relu=False)
        self.add_layer = layers.Add(name='add_%s' %(block_id))
        self.relu_output = layers.Activation(activation='relu', name='relu_%s' %(block_id))

    def call(self, inputs, training):
        layer1 = self.relu_layer_1(self.bn_layer_1(self.conv_layer_1(inputs), training=training))
        layer2 = self.bn_layer_2(self.conv_layer_2(layer1), training=training)
        add = self.add_layer([inputs, layer2])
        outputs = self.relu_output(add)
        return outputs

"""ResNet component for L3."""
class resnet_block_nonuniform_kernals(base_model):
    def __init__(self, seq_length, n_channels, n_classes, block_id):
        super(resnet_block_nonuniform_kernals, self).__init__(seq_length, n_channels, n_classes)
        self.conv_layer_1, self.bn_layer_1, self.relu_layer_1 = self._conv_layer(
                filters=200, kernal_size=7,
                layer_id="%s_layer_%d" %(block_id,1), use_relu=True)
        self.conv_layer_2, self.bn_layer_2, self.relu_layer_2 = self._conv_layer(
                filters=200, kernal_size=3,
                layer_id="%s_layer_%d" %(block_id,2), use_relu=True)
        self.conv_layer_3, self.bn_layer_3 = self._conv_layer(
                filters=200, kernal_size=3,
                layer_id="%s_layer_%d" %(block_id,3), use_relu=False)
        self.add_layer = layers.Add(name='add_%s' %(block_id))
        self.relu_output = layers.Activation(activation='relu', name='relu_%s' %(block_id))

    def call(self, inputs, training):
        layer1 = self.relu_layer_1(self.bn_layer_1(self.conv_layer_1(inputs), training=training))
        layer2 = self.relu_layer_2(self.bn_layer_2(self.conv_layer_2(layer1), training=training))
        layer3 = self.bn_layer_3(self.conv_layer_3(layer2), training=training)
        add = self.add_layer([inputs, layer3])
        outputs = self.relu_output(add)
        return outputs

"""ResNet adapted from ChromDragoNN.

ChromDragoNN was implemented in pytorch and published in Surag Nair, et al, Bioinformatics, 2019.
The original code can be found in https://github.com/kundajelab/ChromDragoNN

This ResNet consists of:
    - 2 convolutional layers --> 128 channels, filter size (5,1)
    - 2 convolutional layers --> 256 channels, filter size (3,1)
    - 1 convolutional layers --> 64 channels, filter size (1,1)
    - 2 x L1Block
    - 1 conv layer
    - 2 x L2Block
    - maxpool
    - 1 conv layer
    - 2 x L3Block
    - maxpool
    - 2 x L4Block
    - 1 conv layer
    - maxpool
    - 2 fully connected layers

L1Block: 2 convolutional layers, 64 channels, filter size (3,1)
L2Block: 2 convolutional layers, 128 channels, filter size (7,1)
L3Block: 3 convolutional layers, 200 channels, filter size (7,1), (3,1),(3,1)
L4Block: 2 convolutional layers, 200 channels, filter size (7,1)
"""
class resnet_model(base_model):
    def __init__(self, seq_length, n_channels, n_classes):
        super(resnet_model, self).__init__(seq_length, n_channels, n_classes)
        self.conv_layer_1, self.bn_layer_1, self.relu_layer_1 = self._conv_layer(
                filters=128, kernal_size=5, layer_id="main_1", use_relu=True)
        self.conv_layer_2, self.bn_layer_2, self.relu_layer_2 = self._conv_layer(
                filters=128, kernal_size=5, layer_id="main_2", use_relu=True)
        self.conv_layer_3, self.bn_layer_3, self.relu_layer_3 = self._conv_layer(
                filters=256, kernal_size=3, layer_id="main_3", use_relu=True)
        self.conv_layer_4, self.bn_layer_4, self.relu_layer_4 = self._conv_layer(
                filters=256, kernal_size=3, layer_id="main_4", use_relu=True)
        self.conv_layer_5, self.bn_layer_5, self.relu_layer_5 = self._conv_layer(
                filters=64, kernal_size=1, layer_id="main_5", use_relu=True)

        self.block_1_1 = resnet_block_uniform_kernels(
                seq_length=seq_length, n_channels=n_channels, n_classes=n_classes,
                filters=64, kernal_size=3, block_id="block_1_1")
        self.block_1_2 = resnet_block_uniform_kernels(
                seq_length=seq_length, n_channels=n_channels, n_classes=n_classes,
                filters=64, kernal_size=3, block_id="block_1_2")
        self.conv_block_1_out, self.bn_block_1_out, self.relu_block_1_out = self._conv_layer(
                filters=128, kernal_size=3, layer_id="block_1_out", use_relu=True)

        self.block_2_1 = resnet_block_uniform_kernels(
                seq_length=seq_length, n_channels=n_channels, n_classes=n_classes,
                filters=128, kernal_size=7, block_id="block_2_1")
        self.block_2_2 = resnet_block_uniform_kernels(
                seq_length=seq_length, n_channels=n_channels, n_classes=n_classes,
                filters=128, kernal_size=7, block_id="block_2_2")
        self.maxpool_block_2 = layers.MaxPool1D(pool_size=3, padding='same', name="block_2_maxpool")
        self.conv_block_2_out, self.bn_block_2_out, self.relu_block_2_out = self._conv_layer(
                filters=200, kernal_size=1, layer_id="block_2_out", use_relu=True)

        self.block_3_1 = resnet_block_nonuniform_kernals(
                seq_length=seq_length, n_channels=n_channels, n_classes=n_classes,
                block_id="block_3_1")
        self.block_3_2 = resnet_block_nonuniform_kernals(
                seq_length=seq_length, n_channels=n_channels, n_classes=n_classes,
                block_id="block_3_2")
        self.maxpool_block_3 = layers.MaxPool1D(pool_size=4, padding='same', name="block_3_maxpool")

        self.block_4_1 = resnet_block_uniform_kernels(
                seq_length=seq_length, n_channels=n_channels, n_classes=n_classes,
                filters=200, kernal_size=7, block_id="block_4_1")
        self.block_4_2 = resnet_block_uniform_kernels(
                seq_length=seq_length, n_channels=n_channels, n_classes=n_classes,
                filters=200, kernal_size=7, block_id="block_4_2")
        self.conv_block_4_out, self.bn_block_4_out, self.relu_block_4_out = self._conv_layer(
                filters=200, kernal_size=7, layer_id="block_4_out", use_relu=True)
        self.maxpool_block_4 = layers.MaxPool1D(pool_size=4, padding='same', name="block_4_maxpool")
        self.flatten_layer = layers.Flatten(name="flatten")
        
        self.fc_1, self.bn_fc_layer_1, self.relu_fc_layer_1 = self._fc_layer(
                units=1000, layer_id="main_fc_1", post_processing=True)
        self.dropout_1 = layers.Dropout(rate=0.3, name="dropout_1")
        self.fc_2, self.bn_fc_layer_2, self.relu_fc_layer_2 = self._fc_layer(
                units=1000, layer_id="main_fc_2", post_processing=True)
        self.dropout_2 = layers.Dropout(rate=0.3, name="dropout_2")
        self.fc_3 = self._fc_layer(units=n_classes, layer_id="main_fc_3", post_processing=False)
        self.sigmoid = layers.Activation(activation='sigmoid', name='sigmoid')

    def call(self, inputs, training):
        layer1 = self.relu_layer_1(self.bn_layer_1(self.conv_layer_1(inputs), training=training))
        layer2 = self.relu_layer_2(self.bn_layer_2(self.conv_layer_2(layer1), training=training))
        layer3 = self.relu_layer_3(self.bn_layer_3(self.conv_layer_3(layer2), training=training))
        layer4 = self.relu_layer_4(self.bn_layer_4(self.conv_layer_4(layer3), training=training))
        layer5 = self.relu_layer_5(self.bn_layer_5(self.conv_layer_5(layer4), training=training))
        block_1_1 = self.block_1_1(inputs=layer5, training=training)
        block_1_2 = self.block_1_2(inputs=block_1_1, training=training)
        block_1_out = self.relu_block_1_out(self.bn_block_1_out(self.conv_block_1_out(block_1_2), training=training))
        block_2_1 = self.block_2_1(inputs=block_1_out, training=training)
        block_2_2 = self.block_2_2(inputs=block_2_1, training=training)
        block_2_maxpool = self.maxpool_block_2(block_2_2)
        block_2_out = self.relu_block_2_out(self.bn_block_2_out(self.conv_block_2_out(block_2_maxpool), training=training))
        block_3_1 = self.block_3_1(inputs=block_2_out, training=training)
        block_3_2 = self.block_3_2(inputs=block_3_1, training=training)
        block_3_maxpool = self.maxpool_block_3(block_3_2)
        block_4_1 = self.block_4_1(inputs=block_3_maxpool, training=training)
        block_4_2 = self.block_4_2(inputs=block_4_1, training=training)
        block_4_out = self.relu_block_4_out(self.bn_block_4_out(self.conv_block_4_out(block_4_2), training=training))
        block_4_maxpool = self.maxpool_block_4(inputs=block_4_out, training=training)
        flattened = self.flatten_layer(block_4_maxpool)
        fc_1 = self.relu_fc_layer_1(self.bn_fc_layer_1(self.fc_1(flattened), training=training))
        dropout_1 = self.dropout_1(inputs=fc_1, training=training)
        fc_2 = self.relu_fc_layer_2(self.bn_fc_layer_2(self.fc_2(dropout_1), training=training))
        dropout_2 = self.dropout_2(inputs=fc_2, training=training)
        fc_3 = self.fc_3(dropout_2)
        outputs = self.sigmoid(fc_3)
        return outputs

"""Execute model fitting and evalation."""
class model_runner:
    def __init__(self, batch_size, seq_length, n_channels, n_classes):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_channels = n_channels
        self.n_classes = n_classes

    def train(self, filename_train, n_samples_train, filename_valid, n_samples_valid,
            model_save_path, learning_rate, min_lr, n_epochs, n_epochs_per_iteration, saved_model=None):
        if tf.io.gfile.exists(model_save_path):
            tf.io.gfile.rmtree(model_save_path)
        tf.io.gfile.makedirs(model_save_path)

        dataset_train = self._load_data(filename_train, n_samples_train, n_epochs=None, shuffle=True)
        dataset_valid = self._load_data(filename_valid, n_samples_valid, n_epochs=1, shuffle=False)
        if not saved_model:
            print ("Initialize a new model.")
        else:
            print ("Loading model from %s." %saved_model)
        model = self.build(saved_model=saved_model, training=True)
        optimizer = self._get_optimizer(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[self._get_lr_metric(optimizer)])
        earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=25)
        reduceLROnPlateau = callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.1, patience=10, min_delta=0.0001, min_lr=min_lr)
        modelCheckpoint = callbacks.ModelCheckpoint(filepath=os.path.join(model_save_path, "model.ckpt"),
                monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
        history = callbacks.History()
        steps_per_epoch = int(n_samples_train/float(self.batch_size * n_epochs_per_iteration))
        model.fit(x=dataset_train, batch_size=self.batch_size, epochs=n_epochs,
                steps_per_epoch=steps_per_epoch, verbose=1,
                callbacks=[earlyStopping, history, reduceLROnPlateau, modelCheckpoint],
                validation_data=dataset_valid, use_multiprocessing=True)

        history_path = os.path.join(model_save_path, "history.txt")
        with open(history_path, 'w') as ofile:
            for name in history.history:
                ofile.write("%s\n" %(name))
                for val in history.history[name]:
                    ofile.write("%f\t" %(val))
                    ofile.write("\n")

    def fine_tune(self, filename_train, n_samples_train, filename_valid, n_samples_valid,
            model_save_path, learning_rate, min_lr, n_epochs, n_epochs_per_iteration,
            pretrained_model, terminal_transfer_layer, lr_reduce_patience=10):
        if tf.io.gfile.exists(model_save_path):
            tf.io.gfile.rmtree(model_save_path)
        tf.io.gfile.makedirs(model_save_path)

        dataset_train = self._load_data(filename_train, n_samples_train, n_epochs=None, shuffle=True)
        dataset_valid = self._load_data(filename_valid, n_samples_valid, n_epochs=1, shuffle=False)
        print ("Loading model from %s till %s layer." %(pretrained_model, terminal_transfer_layer))
        model = self._load_from_model(saved_model=pretrained_model, terminal_layer=terminal_transfer_layer)
        optimizer = self._get_optimizer(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[self._get_lr_metric(optimizer)])

        earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=25)
        reduceLROnPlateau = callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.1, patience=lr_reduce_patience, min_delta=0.0001, min_lr=min_lr)
        modelCheckpoint = callbacks.ModelCheckpoint(filepath=os.path.join(model_save_path, "model.ckpt"),
                monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
        history = callbacks.History()
        steps_per_epoch = int(n_samples_train/float(self.batch_size * n_epochs_per_iteration))
        model.fit(x=dataset_train, batch_size=self.batch_size, epochs=n_epochs,
                steps_per_epoch=steps_per_epoch, verbose=1,
                callbacks=[earlyStopping, history, reduceLROnPlateau, modelCheckpoint],
                validation_data=dataset_valid, use_multiprocessing=True)

        history_path = os.path.join(model_save_path, "history.txt")
        with open(history_path, 'w') as ofile:
            for name in history.history:
                ofile.write("%s\n" %(name))
                for val in history.history[name]:
                    ofile.write("%f\t" %(val))
                    ofile.write("\n")

    def build(self, saved_model, training):
        model = resnet_model(self.seq_length, self.n_channels, self.n_classes)
        x = tf.keras.Input(shape=(self.seq_length, self.n_channels))
        model_built = tf.keras.Model(inputs=x, outputs=model.call(x, training=training))
        if saved_model is not None:
            model_built.load_weights(os.path.join(saved_model, "model.ckpt")).expect_partial()
        return model_built

    def evaluate(self, results_save_path, model_path, filename_test, n_samples_test):
        if tf.io.gfile.exists(results_save_path):
            tf.io.gfile.rmtree(results_save_path)
        tf.io.gfile.makedirs(results_save_path)

        dataset_test = self._load_data(filename_test, n_samples_test, n_epochs=1, shuffle=False)
        seqs, labels = [], []
        for (x, y) in dataset_test:
            seqs.append(x)
            labels.append(y)
        seqs, labels = np.concatenate(seqs), np.concatenate(labels)

        model = self.build(saved_model=model_path, training=False)
        model.compile(optimizer=self._get_optimizer(), loss=tf.keras.losses.BinaryCrossentropy())
        print (model.evaluate(seqs, labels))
        predictions = np.array(model.predict(seqs))

        summary_file = "%s/auc_summary.txt" %(results_save_path)
        for feature_index in range(self.n_classes):
            lbl = labels[:, feature_index]
            lgt = predictions[:, feature_index]
            self._save_predictions("%s/category_%d" %(results_save_path, feature_index), lbl, lgt)
            self._save_auc(summary_file, lbl, lgt, feature_index)

    def annotate(self, results_save_path, model_path, filename_vis, n_samples_vis):
        if tf.io.gfile.exists(results_save_path):
            tf.io.gfile.rmtree(results_save_path)
        tf.io.gfile.makedirs(results_save_path)
        model = self.build(saved_model=model_path, training=False)
        grad_model = tf.keras.models.Model([model.inputs],
                [model.get_layer("conv_main_4").output, model.get_layer("fc_main_fc_3").output])

        dataset_vis = self._load_data(filename_vis, n_samples_vis, n_epochs=1, shuffle=False)
        seqs, labels = [], []
        scores_list = []
        _index = 0
        for (seqs_batch, labels_batch) in dataset_vis:
            for seq in seqs_batch:
                _index += 1
                if (_index % 1000) == 0:
                    print (_index)
                seq_model_input = tf.Variable([seq], dtype=float)
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(seq_model_input)
                conv_grads, seq_grads = tape.gradient(predictions, [conv_outputs, seq_model_input])
                conv_outputs_val, conv_grads_val, seq_grads_val = \
                        (conv_outputs.numpy()[0], conv_grads.numpy()[0], seq_grads.numpy()[0])

                # cam layer
                cam = np.zeros(conv_outputs_val.shape[0], dtype=np.float32)
                filter_weights = np.mean(conv_grads_val, axis=0)
                for i, w in enumerate(filter_weights):
                    cam += (w * conv_outputs_val[:, i])

                # saliency map
                seq_val = seq.numpy()
                saliency_map = np.zeros(seq_val.shape[0], dtype=np.float32)
                for i in range(seq_val.shape[1]-1): # excluding the dimension for coordinates
                    saliency_map += (seq_val[:, i] * seq_grads_val[:, i])

                gradcam = cam * saliency_map
                scores_list.append(gradcam)
                #gradcam_sum = sum([abs(scr) for scr in gradcam])
                #if gradcam_sum == 0:
                #    gradcam_norm = gradcam
                #else:
                #    gradcam_norm = [scr/gradcam_sum for scr in gradcam]
                #scores_list.append(gradcam_norm)
        self._save_annotation_scores(results_save_path, scores_list)

    def annotate_experiment(self, results_save_path, model_path, filename_vis, n_samples_vis):
        if tf.io.gfile.exists(results_save_path):
            tf.io.gfile.rmtree(results_save_path)
        tf.io.gfile.makedirs(results_save_path)
        tf.io.gfile.makedirs(os.path.join(results_save_path, "seq_onehot"))
        tf.io.gfile.makedirs(os.path.join(results_save_path, "coord"))

        model = self.build(saved_model=model_path, training=False)
        grad_model = tf.keras.models.Model([model.inputs],
                [model.get_layer("conv_main_4").output, model.get_layer("fc_main_fc_3").output])

        dataset_vis = self._load_data(filename_vis, n_samples_vis, n_epochs=1, shuffle=False)
        seqs, labels = [], []
        scores_list = []
        scores_coord_list = []
        _index = 0
        for (seqs_batch, labels_batch) in dataset_vis:
            for seq in seqs_batch:
                _index += 1
                if (_index % 1000) == 0:
                    print (_index)
                    break
                seq_model_input = tf.Variable([seq], dtype=float)
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(seq_model_input)
                conv_grads, seq_grads = tape.gradient(predictions, [conv_outputs, seq_model_input])
                conv_outputs_val, conv_grads_val, seq_grads_val = \
                        (conv_outputs.numpy()[0], conv_grads.numpy()[0], seq_grads.numpy()[0])

                # cam layer
                cam = np.zeros(conv_outputs_val.shape[0], dtype=np.float32)
                filter_weights = np.mean(conv_grads_val, axis=0)
                for i, w in enumerate(filter_weights):
                    cam += (w * conv_outputs_val[:, i])

                # saliency map
                seq_val = seq.numpy()
                saliency_map = np.zeros(seq_val.shape[0], dtype=np.float32)
                for i in range(seq_val.shape[1]-1): # excluding the dimension for coordinates
                    saliency_map += (seq_val[:, i] * seq_grads_val[:, i])

                #print (sum([abs(scr) for scr in saliency_map])/len(saliency_map))
                #print (sum([abs(scr) for scr in seq_grads_val[:, 4]])/len(seq_grads_val[:, 4]))

                #gradcam = cam * saliency_map
                gradcam = saliency_map
                gradcam_sum = sum([abs(scr) for scr in gradcam])
                if gradcam_sum == 0:
                    gradcam_norm = gradcam
                else:
                    gradcam_norm = [scr/gradcam_sum for scr in gradcam]
                scores_list.append(gradcam_norm)

                coord_grad = seq_grads_val[:, 4]
                coord_grad_sum = sum([abs(scr) for scr in coord_grad])
                if coord_grad_sum == 0:
                    coord_grad_norm = coord_grad
                else:
                    coord_grad_norm = [scr/coord_grad_sum for scr in coord_grad]
                scores_coord_list.append(coord_grad_norm)
            break
        #self._save_annotation_scores(results_save_path, scores_list)
        self._save_annotation_scores(os.path.join(results_save_path, "seq_onehot"), scores_list)
        self._save_annotation_scores(os.path.join(results_save_path, "coord"), scores_coord_list)

    def print_weights(self, model_path, filename_test, n_samples_test):
        """This function is mainly used for debugging and printing
        weights in selected layers.
        """
        dataset_test = self._load_data(filename_test, n_samples_test, n_epochs=1, shuffle=False)
        seqs, labels = [], []
        for (x, y) in dataset_test:
            seqs.append(x)
            labels.append(y)
        seqs, labels= np.concatenate(seqs), np.concatenate(labels)

        if isinstance(model_path, str):
            model = resnet_model(self.seq_length, self.n_channels, self.n_classes)
            model.compile(optimizer=self._get_optimizer(), loss=tf.keras.losses.BinaryCrossentropy())
            model.load_weights(os.path.join(model_path, "model.ckpt")).expect_partial()
        else:
            model = model_path
        print (model.evaluate(seqs, labels))

        #########
        # a few examples
        #########
        #index = 0
        #while model.get_layer(index=index).name != "flatten":
        #    print(model.get_layer(index=index).name)
        #    print(model.get_layer(index=index).weights)
        #    index += 1
        #print(model.get_layer(name="conv_block_4_out").weights)
        #print(model.get_layer(name="fc_main_fc_2").get_weights())

    def _load_data(self, filename, n_samples, n_epochs, shuffle):
        loader = data_loader(batch_size=self.batch_size, seq_size=self.seq_length,
                n_channels=self.n_channels, n_classes=self.n_classes)
        dataset = loader.load(filename=filename, n_samples=n_samples, n_epochs=n_epochs, shuffle=shuffle)
        return dataset

    def _load_from_model(self, saved_model, terminal_layer):
        """Load weights from 'saved_model' to the current model.
        The weight loading range includes layers from the first layer to the 'terminal_layer'.
        """
        current_model = self.build(saved_model=None, training=True)
        layer_index = 0
        while layer_index < 9999:
            current_layer = current_model.layers[layer_index]
            layer_to_load = saved_model.get_layer(index=layer_index)
            current_layer.set_weights(layer_to_load.get_weights())
            if current_layer.name == terminal_layer:
                break
            else:
                layer_index += 1
        return current_model

    def _get_optimizer(self, learning_rate=1e-3):
        #return tfa.optimizers.MovingAverage(tf.keras.optimizers.Adam(learning_rate=learning_rate))
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _get_lr_metric(self, optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr

    def _save_auc(self, summary_file, lbl, lgt, feature_index):
        try:
            auc_value = roc_auc_score(lbl, lgt)
        except:
            auc_value = -1
        try:
            pr_value = average_precision_score(lbl, lgt)
        except:
            pr_value = -1
        with open(summary_file,'a') as ofile:
            line_to_save = "%d\t%f\t%f\n" %(feature_index, auc_value, pr_value)
            ofile.write(line_to_save)

    def _save_predictions(self, results_save_path, labels, predictions):
        if tf.io.gfile.exists(results_save_path):
            tf.io.gfile.rmtree(results_save_path)
        tf.io.gfile.makedirs(results_save_path)
        file_path = "%s/labels_logits.txt" %(results_save_path)
        with open(file_path, 'w') as ofile:
            for sample_index in range(len(labels)):
                line_to_save = "%f\t%f\n" %(labels[sample_index], predictions[sample_index])
                ofile.write(line_to_save)

    def _save_annotation_scores(self, results_save_path, scores_list):
        with open(os.path.join(results_save_path, "scores.txt"), 'w') as ofile:
            for scores in scores_list:
                line_to_write = ";".join([str(scr) for scr in scores])
                line_to_write += "\n"
                ofile.write(line_to_write)




# unit test
if __name__ == "__main__":
    model = resnet_model(seq_length=1000, n_channels=4, n_classes=919).build_model(input_shape=(1000, 4))
    model.summary()
    """
    pretrain_experiment = model_runner(batch_size=5, seq_length=1000, n_channels=4, n_classes=919)
    filename_train = "/storage/pandaman/project/Alzheimers_ResNet/storage/pretrain/seqs_one_hot/training/data.txt"
    filename_valid = "/storage/pandaman/project/Alzheimers_ResNet/storage/pretrain/seqs_one_hot/validation/data.txt"
    model_save_path = "/storage/pandaman/project/Alzheimers_ResNet/storage/debug/models_debug"
    pretrain_experiment.train(filename_train=filename_train, n_samples_train=1000,
            filename_valid=filename_valid, n_samples_valid=1000, model_save_path=model_save_path,
            learning_rate=2e-3, min_lr=1e-4, n_epochs=2, n_epochs_per_iteration=20)
    model_save_path_new = "/storage/pandaman/project/Alzheimers_ResNet/storage/debug/models_debug_continued"
    pretrain_experiment.train(filename_train=filename_train, n_samples_train=1000,
            filename_valid=filename_valid, n_samples_valid=1000, model_save_path=model_save_path_new,
            learning_rate=2e-4, min_lr=1e-4, n_epochs=2, n_epochs_per_iteration=20, saved_model=model_save_path)
    results_save_path = "/storage/pandaman/project/Alzheimers_ResNet/storage/debug/results_debug"
    pretrain_experiment.evaluate(results_save_path, model_save_path_new, filename_valid, 1000)

    model = resnet_model(seq_length=1000, n_channels=4, n_classes=1).build_model(input_shape=(1000, 4))
    model.summary()
    pretrained_model = pretrain_experiment.build(saved_model=model_save_path_new, training=True)
    experiment = model_runner(batch_size=5, seq_length=1000, n_channels=4, n_classes=1)
    filename_train_ft = "/storage/pandaman/project/Alzheimers_ResNet/storage/experiments/seqs_one_hot/PU1_H3K27Ac/training/data.txt"
    filename_valid_ft = "/storage/pandaman/project/Alzheimers_ResNet/storage/experiments/seqs_one_hot/PU1_H3K27Ac/validation/data.txt"
    fine_tuned_save_path = "/storage/pandaman/project/Alzheimers_ResNet/storage/debug/models_fine_tuned_debug"
    experiment.fine_tune(filename_train=filename_train_ft, n_samples_train=34204,
            filename_valid=filename_valid_ft, n_samples_valid=2550, model_save_path=fine_tuned_save_path,
            learning_rate=2e-3, min_lr=1e-4, n_epochs=2, n_epochs_per_iteration=20,
            pretrained_model=pretrained_model, terminal_transfer_layer="flatten")
    results_fine_tuned_path = "/storage/pandaman/project/Alzheimers_ResNet/storage/debug/results_fine_tuned_debug"
    experiment.evaluate(results_fine_tuned_path, fine_tuned_save_path, filename_valid_ft, 2550)
    experiment.annotate(os.path.join(results_fine_tuned_path, "test"), fine_tuned_save_path, filename_vis=filename_valid_ft, n_samples_vis=2550)
    """
