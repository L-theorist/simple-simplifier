from tensorflow.keras import Input, models, layers, callbacks, optimizers
import nltk
import numpy as np
import simplifier

def build_model(vocab_in, vocab_out, length_in, length_out, n_units, use_emb=None, unfreeze_emb=True):
    if use_emb is not None:
        assert n_units==use_emb.shape[-1], "Embedding dimension should match n_units."
    encoder_input = Input(shape=(length_in,))
    encoder_output = layers.Embedding(vocab_in+1,
                                    n_units,
                                    input_length=length_in,
                                    embeddings_initializer='lecun_uniform',
                                    mask_zero=True,
                                    trainable=True)(encoder_input)
    encoder_output = layers.LSTM(n_units)(encoder_output)
    encoder_output = layers.RepeatVector(length_out)(encoder_output)
    decoder_output = layers.LSTM(n_units, return_sequences=True)(encoder_output)
    decoder_output = layers.TimeDistributed(layers.Dense(vocab_out+1, activation='softmax'))(decoder_output)

    model = models.Model(encoder_input, decoder_output)
    if use_emb is not None:
        model.layers[1].set_weights([use_emb])
        model.layers[1].trainable = unfreeze_emb




    return model
def build_ed_model(vocab_in, vocab_out, length_in, length_out, n_units, use_emb=None, unfreeze_emb=True):
    if use_emb is not None:
        assert n_units==use_emb.shape[-1], "Embedding dimension should match n_units."
    #Encoder
    encoder_input = Input(shape=(None,))# vocab_in+1))
    encoder_emb = layers.Embedding(vocab_in+1,
                                    n_units,
                                    #input_length=length_in,
                                    #embeddings_initializer='lecun_uniform',
                                    mask_zero=True,
                                    #trainable=True
                                    )(encoder_input)
    encoder_lstm = layers.LSTM(n_units, return_state=True)
    #encoder_output = layers.RepeatVector(length_out)(encoder_output)
    encoder_output, state_h, state_c = encoder_lstm(encoder_emb)
    encoder_states = [state_h, state_c]
    #Decoder
    decoder_input = Input(shape=(None,))# vocab_out+1))
    decoder_emb_layer = layers.Embedding(vocab_out , n_units, mask_zero=True)
    decoder_emb = decoder_emb_layer(decoder_input)
    decoder_lstm = layers.LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=encoder_states)
    #decoder_output = layers.TimeDistributed(layers.Dense(vocab_out+1, activation='softmax'))(decoder_output)

    decoder_dense = layers.Dense(vocab_out, activation='softmax')
    decoder_output = decoder_dense(decoder_outputs)

    model = models.Model([encoder_input, decoder_input], decoder_output)
    if use_emb is not None:
        model.layers[1].set_weights([use_emb])
        model.layers[1].trainable = unfreeze_emb
    # Encoder&Decoder for predictions
    encoder_model = models.Model(encoder_input, encoder_states)
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_emb2 = decoder_emb_layer(decoder_input)

    decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_emb2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    decoder_model = models.Model(
                [decoder_input] + decoder_states_inputs,
                [decoder_outputs2] + decoder_states2)


    return model, encoder_model, decoder_model


def decode_sequence(sequence, enc, dec, token_dict, token_dict_reverse=None, length_out=50):
    if token_dict_reverse is None:
        token_dict_reverse = dict((index, word) for word, index in token_dict.items())
        token_dict_reverse[0] = '<pad>'
    # Encode the input as state vectors.
    #states_value = encoder_model.predict(sequence)
    states_value = enc.predict(sequence)

    # Generate empty target sequence of length 1.
    decoded_sequence = np.zeros((1,1))

    # Populate the first character of target sequence with the start character.
    decoded_sequence[0, 0] = token_dict['<START>']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        #output_tokens, h, c = decoder_model.predict([decoded_sequence] + states_value)
        output_tokens, h, c = dec.predict([decoded_sequence] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = token_dict_reverse[sampled_token_index]
        decoded_sentence += ' '+sampled_word

        # Exit condition: either hit max length or find stop token.
        if (sampled_word == '<END>' or len(decoded_sentence) > length_out):
            stop_condition = True

        # Update the target sequence (of length 1).
        decoded_sequence = np.zeros((1,1))
        decoded_sequence[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


def bleu_score(ref, hyp):

    ref = simplifier.tokenize(ref[0])
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([ref], hyp, weights=(1, 1, 1, 1))
    return np.round(BLEUscore*100,1)
