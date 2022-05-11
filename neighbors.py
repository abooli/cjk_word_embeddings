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
    
    words = [x.strip() for x in args.file]
    fp = open(args.output, "w", encoding = "utf8")

    for word in words:
        print("The {} closest words to {} are:".format(10, word))
        word_vector = embedding.get_vec(word, word_list, vectors)
        closest = closest_vectors(word_vector, word_list, vectors, 10)
        for similarity, similar_word in closest:
            print("    * {} (similarity {})".format(similar_word, similarity))
            fp.write("    * {} (similarity {})\n".format(similar_word, similarity))
        print()

    
    fp.close()
        
    # print(top_n_neighbors(word, word_list, vectors, 10))
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the n closest words to " +\
                      "a given word (if specified) or to all of the " +\
                       "words in a text file (if specified). If " +\
                       "neither is specified, compute nothing.")
    
    parser.add_argument("npyFILE",
                        help='an .npy file to read the saved numpy data from')
    parser.add_argument("txtFILE",
                        help='an .txt file to read the saved word list from')
    
    parser.add_argument("--file", "-f", metavar="FILE", type=argparse.FileType('r'),
                        help="a text file with one word per line.")
    parser.add_argument("--output", "-o", help='output file')
    
    args = parser.parse_args()
    main(args)