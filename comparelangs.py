#!/usr/bin/env python3

# import glove
from ast import Import
import numpy
import argparse
import embedding

def compute_length(a):
    a_axis = a.ndim - 1
    array_lengths = numpy.linalg.norm(a, axis=a_axis)
    return array_lengths

def cosine_similarity(array1, array2):
    ## Hint: array2 should be passed into dot first, since we're assuming that array2 will be the (potentially) multidimensional array
    ## Like this:  dot_product = numpy.dot(array2, array1)
    dot_product = numpy.dot(array2, array1)
    length1 = compute_length(array1)
    length2 = compute_length(array2)
    return dot_product / (length1*length2)


def closest_vectors(v, words, array, n):
    similarity = cosine_similarity(v, array)
    result = list(zip(similarity, words))

    result.sort(reverse=True)
    return result[:n]
    

def main(args):
    vectors_lang1 = embedding.load_vectors_array(args.npyFILE_lang1)
    vectors_lang2 = embedding.load_vectors_array(args.npyFILE_lang2)

    word_list = embedding.load_word_list(args.txtFILE)
    
    similarity = []
    for i in range(len(word_list)):
        word = word_list[i]
        word_vector1 = vectors_lang1[i,:]
        word_vector2 = vectors_lang2[i,:]

        
        similarity += [cosine_similarity(word_vector1, word_vector2)]

    result = list(zip(similarity, word_list))
    fp = open(args.output, "w", encoding = "utf8")

    for similarity, word in result:
        fp.write("    * {} (similarity {})\n".format(word, similarity))
        print("    * {} (similarity {})".format(word, similarity))
        print()
        # for similarity, similar_word in closest:
        #    print("    * {} (similarity {})".format(similar_word, similarity))
        #print()
    fp.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the n closest words to " +\
                      "a given word (if specified) or to all of the " +\
                       "words in a text file (if specified). If " +\
                       "neither is specified, compute nothing.")
    
    parser.add_argument("npyFILE_lang1",
                        #type=argparse.FileType('rb'),
                        help='an .npy file to read the saved numpy data from')
    parser.add_argument("npyFILE_lang2",
                        #type=argparse.FileType('rb'),
                        help='an .npy file to read the saved numpy data from')
    
    parser.add_argument("txtFILE",
                        # type=argparse.FileType('wb'),
                        help='an .txt file to read the saved word list from')

    parser.add_argument("--output", "-o", help='output file')

    args = parser.parse_args()
    main(args)
    
