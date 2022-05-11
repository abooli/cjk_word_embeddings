#!/usr/bin/env python3

# import glove
from ast import Import
import numpy
import argparse
import embedding


def main(args):
    vectors = embedding.load_vectors_array(args.npyFILE)
    word_list = embedding.load_word_list(args.txtFILE)
    
    if args.file:
        words = [x.strip() for x in args.file]
    else: return

    vec_list = numpy.zeros((len(words), vectors.shape[1]))

    for i in range(len(words)):
        word = words[i]
        word_vector = embedding.get_vec(word, word_list, vectors)
        vec_list[i,:] = word_vector
    
    print(vec_list)

    if args.listvecFILE:
        embedding.save_vectors_array(vec_list, args.listvecFILE)

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the n closest words to " +\
                      "a given word (if specified) or to all of the " +\
                       "words in a text file (if specified). If " +\
                       "neither is specified, compute nothing.")
    parser.add_argument("--file", "-f", metavar="FILE", type=argparse.FileType('r'),
                        help="a text file with one word per line.")

    parser.add_argument("--listvecFILE", "-v",
                        # type=argparse.FileType('wb'),
                        help='an .npy file to write the saved numpy data to')
    parser.add_argument("npyFILE",
                        #type=argparse.FileType('rb'),
                        help='an .npy file to read the saved numpy data from')
    parser.add_argument("txtFILE",
                        # type=argparse.FileType('wb'),
                        help='an .txt file to read the saved word list from')
    
    args = parser.parse_args()
    main(args)
    
