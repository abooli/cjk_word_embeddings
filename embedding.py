import numpy
import argparse
from re import A

def load_text_vectors(filename):
    fp = open(filename, "r", encoding = "utf8")

    # Read the size of the data
    fp.seek(0)
    header = fp.readline().split()
    rowcount = int(header[0])
    veclength = int(header[1])
    
    # Initialize vectors
    words = []
    vectors = numpy.zeros((rowcount, veclength))
    # print(rowcount, veclength)
    
    # Read in the data
    for (i, v) in enumerate(fp):
        s = v.split(" ")
        print(i, len(s))
        words.append(s[0])
        vectors[i,:] = numpy.array(s[1:])
        # if (i > 2):
            # return words, vectors
        # i += 1
    fp.close()

    return words, vectors

def save_word_list(word_list, filename):
    fp = open(filename, "w", encoding = "utf8")
    fp.writelines("%s\n" % l for l in word_list)
    fp.close()

def save_vectors_array(vectors, filename):
    fp = open(filename, "wb")
    numpy.save(fp, vectors)
    fp.close()

def load_word_list(filename):
    fp = open(filename, "r", encoding="utf-8")
    text = fp.read()
    words = text.split()
    return words

def load_vectors_array(filename):
    fp = open(filename, "rb")
    array = numpy.load(fp, allow_pickle=True)
    return array

def get_vec(word, word_list, vectors):
    index = word_list.index(word)
    vector = vectors[index]
    return vector


def main(args): 
    words, vectors = load_text_vectors(args.vectorFILE)
    save_word_list(words, args.txtFILE)
    save_vectors_array(vectors, args.npyFILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vectorFILE",
                        # type=argparse.FileType('r'),
                        help="a GloVe text file to read from")
    parser.add_argument("npyFILE",
                        # type=argparse.FileType('wb'),
                        help='an .npy file to write the saved numpy data to')
    parser.add_argument("txtFILE",
                        # type=argparse.FileType('wb'),
                        help='an .txt file to write the saved numpy data to')

    args = parser.parse_args()
    main(args)