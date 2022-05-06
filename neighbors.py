import embedding
from similarity import *
import argparse
import numpy as np

def top_n_neighbors(word, word_list, vectors, n):
    vec = embedding.get_vec(word, word_list, vectors)
    pairs = closest_vectors(vec, word_list, vectors, n)
    return pairs

        
def main(args):
    word_list = embedding.load_word_list(args.txtFILE)
    vectors = embedding.load_vectors_array(args.npyFILE)

    word = input()
    print(top_n_neighbors(word, word_list, vectors, 10))
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the n closest words to " +\
                      "a given word (if specified) or to all of the " +\
                       "words in a text file (if specified). If " +\
                       "neither is specified, compute nothing.")
    # parser.add_argument("npyFILE",
                        # type=argparse.FileType('rb'),
                        # help='an .npy file to read the saved numpy data from')
    parser.add_argument("npyFILE",
                        # type=argparse.FileType('wb'),
                        help='an .npy file to write the saved numpy data to')
    parser.add_argument("txtFILE",
                        # type=argparse.FileType('wb'),
                        help='an .txt file to write the saved numpy data to')
    #parser.add_argument("file1")
                        # type=argparse.FileType('r'))
    #parser.add_argument("file2")
                        # type=argparse.FileType('r'))
    # parser.add_argument("relationsFILE",
    #                     type=argparse.FileType('r'),
                        # help='a file containing pairs of relations')
    parser.add_argument("--plot", "-p", default="plot.png", help="Name of file to write plot to.")

    args = parser.parse_args()
    main(args)