import os
from optparse import OptionParser
import train


storage_path = ""
def run(storage_path):
    saved_model = os.path.join(storage_path, "pretrain/models_coordconv/")
    pretrain_experiment = train.model_runner(batch_size=256, seq_length=1000, n_channels=5, n_classes=919)
    pretrained_model = pretrain_experiment.build(saved_model=saved_model, training=True)
    data_path = os.path.join(storage_path, "experiments/")
    for line in open(os.path.join(data_path, "table_matrix/table_TFs_extended.txt")):
        (TF_name, f_pos, f_neg, f_vis) = line.strip().split(";")
        filename_train = os.path.join(data_path, "seqs_one_hot_extended_sliding", TF_name, "training/data.txt")
        size_train_data = sum(1 for line in open(filename_train))
        filename_valid = os.path.join(data_path, "seqs_one_hot_extended_sliding", TF_name, "validation/data.txt")
        size_valid_data = sum(1 for line in open(filename_valid))

        model_save_path = os.path.join(data_path, "models_extended_coordconv_sliding", TF_name)
        experiment = train.model_runner(batch_size=256, seq_length=1000, n_channels=5, n_classes=1)
        experiment.fine_tune(filename_train=filename_train, n_samples_train=size_train_data,
                filename_valid=filename_valid, n_samples_valid=size_valid_data, model_save_path=model_save_path,
                learning_rate=2e-4, min_lr=1e-7, n_epochs=2, n_epochs_per_iteration=7,
                pretrained_model=pretrained_model, terminal_transfer_layer="flatten", lr_reduce_patience=1)

        filename_test = os.path.join(data_path, "seqs_one_hot_extended_sliding", TF_name, "test/data.txt")
        size_test_data = sum(1 for line in open(filename_test))
        results_save_path = os.path.join(data_path, "results_extended_coordconv_sliding", TF_name)
        experiment.evaluate(results_save_path=results_save_path, model_path=model_save_path,
                filename_test=filename_test, n_samples_test=size_test_data)

        filename_vis = os.path.join(data_path, "seqs_one_hot_extended_sliding", TF_name, "visualization/data.txt")
        size_vis_data = sum(1 for line in open(filename_vis))
        experiment.annotate(os.path.join(results_save_path, "annotations"), model_save_path,
                filename_vis=filename_vis, n_samples_vis=size_vis_data)

if __name__ == "__main__":
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option('--path', dest='storage_path', default="",
            help='Path to your data folder.')
    (options, args) = parser.parse_args()
    
    if options.storage_path:
        run(options.storage_path)
    else:
        exit("Please specify parameter --path.")
