from sklearn.utils import shuffle
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras import preprocessing
from unicodedata import normalize
import collections
from operator import itemgetter
from tensorflow.keras.utils import to_categorical
import string
def build_word2index(word_list, add_toc=False):
    """
    Takes a list of (unique!) word tokens. Returns a word-index dictionary, last index is assigned to '<UNK>'.
    If add_toc=True, adds a start/end tokens.
    """
    tokenizer_dict = {}
    i = 1
    for word in set(word_list):
        tokenizer_dict[word] = i
        i += 1
    tokenizer_dict["<UNK>"] = i
    if add_toc:
        tokenizer_dict['<START>'] = i+1
        tokenizer_dict['<END>'] = i+2
    return tokenizer_dict

def tokenize(clean_text):
    token_list = nltk.word_tokenize(clean_text)
    return token_list



def build_vocabulary(token_list, threshold=1, with_freq=True):
    """
    Takes a list of tokens (from corpus), counts them with frequencies and returns a list
    of (word, freq) with freq >= threshold, sorted from frequent to rare. If with_freq=False,
    returns a list of words, still sorted by frequency.
    """
    token_list_Counter = collections.Counter(token_list)
    token_list_over_threshold = []
    for key, value in token_list_Counter.items():
        token_list_over_threshold.append((key, value))
    token_list_over_threshold.sort(key=itemgetter(1), reverse=True)
    if with_freq:
        token_list_over_threshold = [(word, count) for (word, count) in token_list_over_threshold if count >= threshold]
    else:
        token_list_over_threshold = [word for (word, count) in token_list_Counter if count >= threshold]
    #token_list_over_threshold = sorted(token_list_over_threshold, key=lambda x: x[0])
    #vocabulary_size = len(token_list_over_threshold) + 1
    return token_list_over_threshold

def clean(sentences):
    """
    Takes the corpus, decodes into UTF-8, tokenizes on white space, puts everything in lowercase, removes punctuation.
    Returns string.
    """
    translator = str.maketrans('', '', string.punctuation)
    clean_txt = ''
    for line in sentences:

        line = normalize('NFD', line).encode('ascii', 'ignore')
        line = line.decode('UTF-8')
        # tokenize on white space
        line = line.translate(translator)
        line = line.split()
        # convert to lowercase
        line = [word.lower() for word in line]
        # remove punctuation from each token
        #line = [word.translate(table) for word in line]
        # remove non-printable chars form each token
        #line = [re_print.sub('', w) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if (word.isalpha() or word.isnumeric())]
        # store as string
        clean_txt += (' '.join(line))
    return clean_txt



def tokenizer(line, add_toc=False):
    """
    Uses nltk RefexpTokenizer to tokenize wrt alphanumerics, normalizes, decodes UTF8, ensures lower case.
    Takes a Sentence(s) string as input.
    Returns a list of token words.
    If add_toc=True, adds start and end tokens.
    """

    tokenizer = RegexpTokenizer(r'\w+')
    line = normalize('NFD', line).encode('ascii', 'ignore')
    line = line.decode('UTF-8')
    tokenized_line = tokenizer.tokenize(line)
    tokenized_line = [word.lower() for word in tokenized_line]
    if add_toc:
        tokenized_line = ['<START>'] + tokenized_line + ['<END>']
    return tokenized_line


def encode(tokenized_line, token_dict, seq_len, pad=True):
    """
    Takes a list of word tokens and translates it into a list of word indices using a dictionary.
    Returns an array of token words, of shape (?, seq_len). Post padding and post truncating is applied.
    """

    encoded_line = []
    for word in tokenized_line:
        encoded_line.append(token_dict.get(word, token_dict["<UNK>"]))
    if pad:
        encoded_line = preprocessing.sequence.pad_sequences(
        [encoded_line],
        maxlen=seq_len,
        padding="post",
        truncating="post")

    return encoded_line



def one_hot_encode_output(encoded_sentences, sentence_length_dst=50,vocabulary_size_dst=32119):      #, vocab_size):
    curr = encoded_sentences        #encoded_sentences
    number_samples = curr.shape[0]
    decoder_targets_one_hot = np.zeros((
            number_samples,
            sentence_length_dst,
            vocabulary_size_dst  # hm... why?
        ))
    for i, sentence in enumerate(curr):
        for t, word in enumerate(sentence):
            decoder_targets_one_hot[i, t, word] = 1
    return decoder_targets_one_hot

# def one_hot_encode_output(encoded_sentences, sentence_length_dst=50,vocabulary_size_dst=32119):      #, vocab_size):
#     for curr in encoded_sentences:         #encoded_sentences
#         number_samples = curr.shape[0]
#         decoder_targets_one_hot = np.zeros((
#                 number_samples,
#                 sentence_length_dst,
#                 vocabulary_size_dst+1  # hm... why?
#             ))
#         for i, sentence in enumerate(curr):
#             for t, word in enumerate(sentence):
#                 decoder_targets_one_hot[i, t, word] = 1
#         yield decoder_targets_one_hot

def one_hot_encode_ed(sent, vocabulary_size):
    encoded = to_categorical(sent, num_classes=vocabulary_size)
    return encoded


def generator(X_data, Y_data, batch_size, samples=30000, shf=True, onehot=True):
    """generates data for the fit_generator
    input X_data, Y_data, batch_size, samples=30000
    shf=False returns the first #samples of X_data, Y_data; otherwise shuffles first.
    onehot=True applies one hot encoding to y_batch
    yields x_batch, y_batch"""
    #from sklearn.utils import shuffle
    if shf:
        x_data, y_data = shuffle(X_data, Y_data, n_samples=samples)
    else:
        x_data = X_data[0:samples]
        y_data = Y_data[0:samples]
    samples_per_epoch = x_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0

    while True:
        x_batch = np.array(x_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y_batch = np.array(y_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
        if onehot:
            y_batch = one_hot_encode_output(y_batch)
        counter += 1
        yield x_batch,y_batch

        #restart counter to yield data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0

def generator_ed(X_data, Y_data, batch_size, vocab_in, vocab_out, samples=30000, shf=True, onehot_dst=True, onehot_src=True):
    """generates data for the fit_generator of an encoder-decoder model
    input X_data, Y_data, batch_size, samples=30000
    shf=False returns the first #samples of X_data, Y_data; otherwise shuffles first.
    onehot_src=True applies one hot encoding to x_batch
    onehot_dst=True applies one hot encoding to y_batch
    yields x_batch, x2_batch, y_batch"""
    #from sklearn.utils import shuffle
    if shf:
        x_data, y_data = shuffle(X_data, Y_data, n_samples=samples)
    else:
        x_data = X_data[0:samples]
        y_data = Y_data[0:samples]
    samples_per_epoch = x_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    pad = np.zeros((batch_size, 1))
    pad_end = np.zeros((batch_size, 1))
    # pad = np.zeros((batch_size,))
    # pad_end = np.zeros((batch_size,))
    while True:
        x_batch = np.array(x_data[batch_size*counter:batch_size*(counter+1)])#.astype('float32')
        y_batch = np.array(y_data[batch_size*counter:batch_size*(counter+1)])#.astype('float32')
        x2_batch = np.concatenate((pad, y_batch[:,:-1]), axis=1)
        #x2_batch = np.concatenate((pad, y_batch), axis=0)
        y_batch = np.concatenate((y_batch[:,1:], pad_end), axis=1)
        #y_batch = np.concatenate((y_batch, pad_end), axis=0)
        if onehot_src:
            x_batch = one_hot_encode_ed(x_batch, vocab_in)
        if onehot_dst:
            y_batch = one_hot_encode_ed(y_batch, vocab_out)
            #x2_batch = one_hot_encode_ed(x2_batch, vocab_out+3)

        counter += 1
        yield [x_batch, x2_batch], y_batch

        #restart counter to yield data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0


def get_word(n, token_dict):
    """
    Returns the work with index n from dictionary token_dict dictionary. Super slow.
    """
    for word, index in token_dict.items():
        if index == n:
            return word
    return None

def convert(preds, token_dict):
    """
    Converts a list of indices into a list of words using token_dict dictionary.
    """
    preds_text = []
    for i in preds:
        temp = []
        for j in range(len(i)):
            t = get_word(i[j], token_dict)
            if j > 0:
                if (t == get_word(i[j-1], token_dict)) or (t == None):
                    temp.append('')
                else:
                    temp.append(t)
            else:
                    if(t == None):
                        temp.append('')
                    else:
                        temp.append(t)

        preds_text.append(' '.join(temp))
    return preds_text


def predictor(model, enc_line):
    pred = model.predict_classes(enc_line)
    pass
