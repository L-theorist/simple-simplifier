from tensorflow.keras import Input, models, layers, callbacks, optimizers


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




#
#
# model = models.Sequential()
# model.add(layers.Embedding(          #word embedding
#     vocabulary_size_source,
#     #embedding_size,
#     200,
#     input_length=sentence_length,
#     embeddings_initializer='lecun_uniform',
#     mask_zero=True
#
# ))
# model.add(layers.LSTM(200))#, return_sequences=True))
# #model.add(layers.LSTM(256)) #
# model.add(layers.RepeatVector(sentence_length_dst))
# #model.add(layers.LSTM(256, return_sequences=True)) #
# model.add(layers.LSTM(200, return_sequences=True))
# model.add(layers.TimeDistributed(layers.Dense(vocabulary_size_dst+1, activation='softmax')))
# #model.add(layers.Dense(vocabulary_size_source+1, activation='softmax'))
#
# #model.layers[0].set_weights([embedding_matrix])
# #model.layers[0].trainable = True #False   #freeze
