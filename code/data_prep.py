'''This code is for preparing training/validation/test data.
Input data are bed files. Output data are one-hot encoded sequences in txt files.
'''

import os
import random
import math
from optparse import OptionParser
from Bio.Seq import Seq
from Bio import SeqIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class dataset:
    def __init__(self, f_pos, f_neg, f_vis, seq_length):
        self.f_pos = f_pos
        self.f_neg = f_neg
        self.f_vis = f_vis
        self.seq_length = seq_length

    def make_dataset(self, data_dir, chromID_valid, chromID_test, reference_dir):
        coordinates_pos, n_pos = self._read_bed_file(self.f_pos, with_offset=True)
        coordinates_neg, n_neg = self._read_bed_file(self.f_neg, with_offset=True)
        coordinates_vis, n_vis = self._read_bed_file(self.f_vis)

        sequences, sequences_vis = self._coordinate_to_seq(
                coordinates_pos, coordinates_neg, coordinates_vis, reference_dir)
        seqs_train, seqs_valid, seqs_test  = [], [], []
        for label in [0, 1]:
            for (seq, label, chromID, start, end, seq_start, seq_end, offset) in sequences[label]:
                if chromID == chromID_valid:
                    seqs_valid.append((seq, label, offset))
                elif chromID == chromID_test:
                    seqs_test.append((seq, label, offset))
                else:
                    seqs_train.append((seq, label, offset))
        self._save_seqs_into_file(dataset=self._prepare_dataset(seqs_train), data_dir=os.path.join(data_dir, "training"))
        self._save_seqs_into_file(dataset=self._prepare_dataset(seqs_valid), data_dir=os.path.join(data_dir, "validation"))
        self._save_seqs_into_file(dataset=self._prepare_dataset(seqs_test), data_dir=os.path.join(data_dir, "test"))

        dataset_vis, auxiliary_info_vis = self._prepare_dataset_vis(sequences_vis)
        self._save_seqs_into_file(dataset=dataset_vis, data_dir=os.path.join(data_dir, "visualization"))
        self._save_aux_info_into_file(dataset=auxiliary_info_vis, data_dir=os.path.join(data_dir, "visualization"))

    def _read_bed_file(self, ifile, with_offset=False):
        data = []
        n_data = 0
        for line in open(ifile):
            elems = line.split()
            chromID, start, end = elems[0], int(elems[1]), int(elems[2])
            if with_offset:
                data.append((chromID, start, end, n_data%10))
            else:
                data.append((chromID, start, end))
            n_data += 1
        return data, n_data

    def _coordinate_to_seq(self, coord_pos, coord_neg, coord_vis, reference_dir):
        genome_limit_path = os.path.join(reference_dir, "hg19.fa.fai")
        reference_genome_dict, chrom_length_dict =\
                self._load_ref_genome(reference_dir, genome_limit_path)

        sequences = [[], []]
        coord = [coord_neg, coord_pos]
        n_out_of_bound = 0
        for label in [0, 1]:
            for chromID, start, end, offset in coord[label]:
                seq_start = start-int((self.seq_length-(end-start))/2.0)
                seq_end = seq_start + self.seq_length
                if (chromID in chrom_length_dict) and \
                        seq_start >= 0 and seq_end < chrom_length_dict[chromID]:
                    seq = (reference_genome_dict[chromID][seq_start:seq_end]).upper()
                    sequences[label].append((seq, label, chromID, start, end, seq_start, seq_end, offset))
                else:
                    n_out_of_bound += 1
        print ("There are %d sequences out of the chromosome boundaries." %(n_out_of_bound))

        sequences_vis = []
        for chromID, start, end in coord_vis:
            seq_start = start-int((self.seq_length-(end-start))/2.0)
            seq_end = seq_start + self.seq_length
            if (chromID in chrom_length_dict) and \
                    seq_start >= 0 and seq_end < chrom_length_dict[chromID]:
                seq = (reference_genome_dict[chromID][seq_start:seq_end]).upper()
                label = 1
                sequences_vis.append((seq, label, chromID, start, end, seq_start, seq_end))
        return sequences, sequences_vis

    def _load_ref_genome(self, reference_dir, genome_limit_path):
        chrom_length_dict = {}
        for line in open(genome_limit_path):
            elems = line.split()
            chromID, length = elems[0], int(elems[1])
            chrom_length_dict[chromID] = length

        ref_dict = {}
        for seq in SeqIO.parse("%s/hg19.fa" %(reference_dir), "fasta"):
            ref_dict[seq.id] = (str(seq.seq)).upper()
        return ref_dict, chrom_length_dict 

    def _prepare_dataset(self, seqs):
        total_seqs = []
        for (seq,label,offset) in seqs:
            total_seqs.append((self._string_to_one_hot(seq, offset, "+"), label))
            total_seqs.append((self._string_to_one_hot(self._reverse_complement(seq), offset, "-"), label))
        random.shuffle(total_seqs)
        return total_seqs

    def _prepare_dataset_vis(self, seqs):
        total_seqs = []
        auxiliary_info = []
        neutral_offset = 4.5
        for (seq, label, chromID, start, end, seq_start, seq_end) in seqs:
            total_seqs.append((self._string_to_one_hot(seq, neutral_offset, "+"), label))
            auxiliary_info.append((chromID, start, end, seq_start, seq_end, "+"))
            total_seqs.append((self._string_to_one_hot(self._reverse_complement(seq), neutral_offset, "-"), label))
            auxiliary_info.append((chromID, start, end, seq_start, seq_end, "-"))
        return total_seqs, auxiliary_info

    def _string_to_one_hot(self, seq, offset, strand):
        data = []
        nuc_idx = 0
        for nuc in seq:
            if strand == "+":
                nuc_offset = float(-50-100*offset+nuc_idx)/500.0
            elif strand == "-":
                nuc_offset = float(950-100*offset-nuc_idx-1)/500.0 
            else:
                exit("Wrong strand symbol: %s" %(strand))
            data.append(self._to_one_hot(nuc, nuc_offset))
            nuc_idx += 1
        data_str = ",".join(data)
        return data_str

    def _to_one_hot(self, nuc, nuc_offset):
        nucleotides = ["A", "T", "C", "G"]
        if nuc == "N":
            return ",".join(["0.25" for _ in range(len(nucleotides))] + [str(nuc_offset)])
        else:
            index = nucleotides.index(nuc)
            onehot = ["0" for _ in range(len(nucleotides))]
            onehot[index] = "1"
            return ",".join(onehot+ [str(nuc_offset)])

    def _reverse_complement(self, seq):
        seq_rc = (Seq(seq)).reverse_complement()
        return str(seq_rc)

    def _save_seqs_into_file(self, dataset, data_dir):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        output_path = os.path.join(data_dir, "data.txt")
        with open(output_path, 'w') as ofile:
            for (seq, label) in dataset:
                line_to_save = "%s;%d\n" %(seq, label)
                ofile.write(line_to_save)

    def _save_aux_info_into_file(self, dataset, data_dir):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        output_path = os.path.join(data_dir, "auxiliary_info.txt")
        with open(output_path, 'w') as ofile:
            for (chromID, start, end, seq_start, seq_end, strand) in dataset:
                line_to_save = "%s,%d,%d,%d,%d,%s\n" %(chromID, start, end, seq_start, seq_end, strand)
                ofile.write(line_to_save)

def main(data_path):
    for line in open(os.path.join(data_path, "table_matrix/table_TFs_extended.txt")):
        (TF_name, f_pos, f_neg, f_vis) = line.strip().split(";")
        f_pos = os.path.join(data_path, f_pos)
        f_neg = os.path.join(data_path, f_neg)
        f_vis = os.path.join(data_path, f_vis)
        ds = dataset(f_pos, f_neg, f_vis, seq_length=1000)

        save_dir = os.path.join(data_path, "seqs_one_hot_extended_sliding", TF_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        reference_dir = os.path.join(data_path, "genomes/hg19/")
        ds.make_dataset(save_dir, chromID_valid="chr8", chromID_test="chr9", reference_dir=reference_dir)

if __name__ == "__main__":
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option('--path', dest='storage_path', default="",
            help='Path to the data folder.')
    (options, args) = parser.parse_args() 

    data_path = os.path.join(options.storage_path, "/experiments/")
    if not options.storage_path:
        main(data_path)
    else:
        exit("Please specify parameter --path.")
###END OF FILE###########################
