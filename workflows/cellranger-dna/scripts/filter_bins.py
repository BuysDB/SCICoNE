import h5py
import argparse
import pandas as pd
import numpy as np

def merge_chromosomes(h5):

    n_cells = h5['cell_barcodes'][:].shape[0]
    all_chromosomes = list(h5['normalized_counts'].keys())
    # list of all cnv arrays
    cnv_matrices = []
    for chr in all_chromosomes:
        cnv_matrices.append(h5['normalized_counts'][chr][:][0:n_cells,:]) # select only the cells, not cell groups

    cell_all_chrs = np.concatenate(cnv_matrices, axis=1)
    return cell_all_chrs

def filter_negative_bins(mat):
    df_arr = pd.DataFrame(mat)
    df_arr[df_arr < 0] = None
    column_filter_mask = ~df_arr.isnull().any()
    return column_filter_mask


parser = argparse.ArgumentParser()
parser.add_argument("-h5", "--hdf5", required=True, help="cellranger-dna hdf5 output")
parser.add_argument("-b", "--bins", required=False, help="list of low quality bins to exclude")
parser.add_argument("-o","--output_path",required=False, default="./", help="path to the output")
parser.add_argument("-s", "--sample_name",required=False, default="", help="name of the sample")

args = parser.parse_args()


# TODO: perform filtering based on the bins

h5f = h5py.File(args.hdf5, 'r')

mat = merge_chromosomes(h5f)

bin_size = h5f["constants"]["bin_size"][()]
n_bins = mat.shape[1]
bin_ids = [x for x in range(0,n_bins)]
bin_df = pd.DataFrame(bin_ids, columns=["bin_ids"])

bin_df["start"] = bin_df["bin_ids"] * bin_size
bin_df["end"] = bin_df["start"] + bin_size
print(bin_df.head())

# all_chromosomes = list(h5f['cnvs'].keys())
# # list of all cnv arrays
# chr_sizes = []
#
# for chr in all_chromosomes:
#     chr_sizes.append(h5f['cnvs'][chr][:].shape[1])
#
# all_chr_ids = []
# for i in range(0,len(chr_sizes)):
#     all_chr_ids += [all_chromosomes[i]]*chr_sizes[i]


if args.bins is None:
    print("no bins to exclude are provided")
else:
    print(args.bins)


negative_bins_filter_mask = filter_negative_bins(mat)

print("negative mask len")
print(len(negative_bins_filter_mask))
print("negative mask sum")
print(sum(negative_bins_filter_mask))

print("matrix shape before & after filtering for negatives")
print(mat.shape)
mat = mat[:,negative_bins_filter_mask]
print(mat.shape)

print("bin_df shape before & after filtering for negatives")
print(bin_df.shape)
bin_df = bin_df[negative_bins_filter_mask]
print(bin_df.shape)

np.savetxt(args.output_path + '/' + args.sample_name +"_filtered_counts.tsv", mat, delimiter='\t')

bin_df.to_csv(args.output_path + '/' + args.sample_name + "_bins_genome.tsv",sep='\t',index=False)

print("Output written to: " + args.output_path)

h5f.close()