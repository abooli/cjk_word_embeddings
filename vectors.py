import gzip
import numpy
import argparse
from re import A

def load_text_vectors(file):
    with gzip.open(file, 'r') as fp:
        fp.seek(0)
        rowcount = 0
        words = []
        for _ in fp:
            rowcount += 1
        vectors = numpy.zeros((rowcount, 300))

        fp.seek(0)
        i = 0
        for v in fp:
            s = v.split()
            words.append(s[0])
            vectors[i,:] = numpy.array(s[1:])
            i += 1       

        return words, vectors


def save_glove_vectors(word_list, vectors, fp):
    numpy.save(fp, vectors)
    numpy.save(fp, word_list)
    fp.close()

def load_glove_vectors(fp):
    array = numpy.load(fp, allow_pickle=True)
    words = list(numpy.load(fp, allow_pickle=True))
    return (array, words)

def get_vec(word, word_list, vectors):
    index = word_list.index(word)
    vector = vectors[index]
    return vector


def main(args): 
    words, vectors = load_text_vectors(args.embeddingFILE)
    save_glove_vectors(words, vectors, args.npyFILE)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddingFILE",
                        # type=argparse.FileType('r'),
                        help="an embedding .gz file to read from")
    parser.add_argument("npyFILE",
                        type=argparse.FileType('wb'),
                        help='an .npy file to write the saved numpy data to')

    args = parser.parse_args()
    main(args)