#!/usr/bin/env python3
from __future__ import print_function
# -*- coding: utf-8 -*-
'''
Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.

Example from: https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
'''

from keras.callbacks import Callback, LambdaCallback, ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import CuDNNGRU, GRU, LSTM, Dropout, Bidirectional
from keras.optimizers import RMSprop, Adam
from keras import regularizers
from keras.utils.data_utils import get_file
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import random
import yaml
import time
import sys
import os
import io

import csv_to_table as cvt

class PlotProgress(Callback):
    def __init__(self, x_size, y_size, chars):
        self.x_size = x_size
        self.y_size = y_size
        self.char_set = chars

        return
    ## __init__##


    def plot_init (self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2,
                  sharex=True,
                  figsize=(10,4))

        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_xlim(0, self.x_size)
        self.ax1.set_ylim(0, self.y_size)
        self.ax1.grid(axis='y', which='major')

        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylim (0, 1)
        self.ax2.grid(axis='y', which='major')

        self.line1 = self.ax1.plot([], [], 'o', color = 'blue')[0]
        self.line2 = self.ax1.plot([], [], '-', color = 'blue')[0]
        self.ax1.legend([self.line1, self.line2], ['Training', 'Validation'])

        self.line3 = self.ax2.plot([], [], 'o', color = 'green')[0]
        self.line4 = self.ax2.plot([], [], '-', color = 'green')[0]
        self.ax2.legend([self.line3, self.line4], ['Training', 'Validation'])

        plt.pause(0.1)

        return self.line1, self.line2, self.line3, self.line4
    ## plot_init ##


    def plot_data (self, loss, acc, val_loss, val_acc, x_as):
        self.line1.set_data(x_as, loss)
        self.line2.set_data(x_as, val_loss)
        self.line3.set_data(x_as, acc)
        self.line4.set_data(x_as, val_acc)

        plt.draw()

        return
    ## plot_data ##


    def on_train_begin(self, logs={}):
        self.plot_init()
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []

        return
    ## on_train_begin ##


    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])
        self.acc.append(logs['accuracy'])
        self.val_acc.append(logs['val_accuracy'])

        # Code to readjust the y_limit of the loss graphics when the losses are
        # greater than the current y_limit
        if self.loss[-1] > self.y_size:
            self.y_size = int(self.loss[-1] + 1)
            self.ax1.set_ylim(0, self.y_size) # max(data[:,1]))
        elif self.val_loss[-1] > self.y_size:
            self.y_size = int(self.val_loss[-1] + 1)
            self.ax1.set_ylim(0, self.y_size) # max(data[:,1]))

        x_as = [i for i in range(len(self.loss))]

        for i in range(epoch):
            self.plot_data (self.loss, self.acc, self.val_loss, self.val_acc, x_as)
            plt.pause(0.1)

        plt.show()

        return
    ## on_epoch_end ##
## Class: PlotProgress ##


class Sequencing():
    def __init__(self, snapshot: str, source:str):
        self.data_source = source
        self.snapshot_dir = snapshot
        self.template = '{:s}_{:s}_{:s}'.format(snapshot, '{:02d}', '{:s}')
        self.count = 1
        filename = self.template.format(self.count, 'log') + '.txt'

        while os.path.exists(filename):
            self.count += 1
            filename = self.template.format(self.count, 'log') + '.txt'

        self.log_path = self.template.format(self.count, 'log') + '.txt'
        self.created_path = self.template.format(self.count, 'created_names') + '.txt'
        self.summ_path = self.template.format(self.count, 'summary') + '.txt'
        self.report_path = self.template.format(self.count, 'generated_names') + '.txt'
        self.model_path = self.template.format(self.count, 'model-l{:03d}-d{:02d}-q{:d}') + '.h5'

        Path(self.log_path).touch()
        Path(self.summ_path).touch()
        Path(self.report_path).touch()
        Path(self.created_path).touch()

        return
    ## __init__ ##


    def read_data (self, source, n_samples, minlen):
        if source == 'nietzsche':
            path = get_file(
                'nietzsche.txt',
                 origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        elif source == 'wiki':
            path = '/media/i/home/data/nl-wiki/nl-wiki-1mb.txt'
        elif source == 'books':
            path = '/media/i/home/data/nl-wiki/books.txt'
        elif source == 'fast':
            path = '/media/i/home/data/nl-wiki/few-books.txt'

        with io.open(path, encoding='utf-8') as f:
            text = f.read().lower()

        print('corpus length:', len(text))

        return [text]
    ## read_data ##


    def create_names(self, n_samples, minlen):
        def find_single(achternaam,):
            namen = []
            i = 0
            while i < voornamen_per_achternaam:
                voor_index = random.randint(0, len(voornamen) - 1)
                voornaam = voornamen.iloc[voor_index].strip().lower()
                naam = ' ' + voornaam + ' ' + achternaam + ' '
                if len(naam) > minlen:
                    namen.append(naam)
                    i += 1

            return namen


        print('Creating names...')
        seconds = time.time()
        random.seed(42)

        converter = cvt.csv_to_table()
        achter_tabel, _, _, _ = converter.read_csv_file('/media/i/home/data/namen/achternamen-1947-2007.csv',
                                                        encoding='ISO-8859-1', sep=';')
        jongens_tabel, _, _, _ = converter.read_csv_file('/media/i/home/data/namen/voornamen-jongens.csv',
                                                        encoding='ISO-8859-1', sep=';')
        meisjes_tabel, _, _, _ = converter.read_csv_file('/media/i/home/data/namen/voornamen-meisjes.csv',
                                                        encoding='ISO-8859-1', sep=';')

        achternamen = achter_tabel['Naam'].str.lower()
        prefixes = achter_tabel['Prefix'].str.lower()
        jongensnamen = (jongens_tabel['Naam'].str.split(' ', 1, expand=True))[0]
        meisjesnamen = (meisjes_tabel['Naam'].str.split(' ', 1, expand=True))[0]
        voornamen = pd.concat([jongensnamen, meisjesnamen])
        voornamen = voornamen.str.lower()

        voornamen_per_achternaam = int(n_samples / len(achternamen))

        # Create a dictionary of prefixes with their frequency. This will be
        # used to generate names with prefixes based on their relative frequency
        prefix_list = list(prefixes.str.lower())
        prefixes = {item:prefix_list.count(item) for item in prefix_list}
        prefixes[''] = prefixes[np.nan] # Empty string becomes NaN, quite irritating
        del prefixes[np.nan] # Replace it by empty text and delete the NaN entry

        # For each surname find voornamen_per_achternaam names by function mapping
        texts = map(find_single, achternamen)

        # a list of lists is returned, flatten the list
        texts = [item for sublist in texts for item in sublist]

        seconds = time.time() - seconds
        cpu_text = '\n\nCreating ' + str(int(len(texts))) + ' names took ' + str(int(seconds)) + ' CPU seconds\n'
        print(cpu_text)

        with open(self.created_path, 'w') as outfile:
            for naam in texts:
                print(naam, file = outfile)

        return texts, prefixes
    ## create_names ##


    def get_data(self, data_source, sequence_lengths):
        maxlen = max(sequence_lengths)
        if data_source == 'names':
            text_data, prefixes = self.create_names(n_samples, maxlen)
        else:
            text_data = self.read_data(self.data_source, n_samples, maxlen)
            prefixes = None

        return text_data, prefixes

    """
    def create_refs(self, texts):
        chars = set()
        for text in texts:
            chars = chars.union(set(text))

        chars = sorted(list(chars))

        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        return chars, char_indices, indices_char
    ## create_refs ##


    def create_sentences(self, texts, sequence_length, step):
        # find out characters and create (reverse) character indices
        chars = set()
        for text in texts:
            chars = chars.union(set(text))

        chars = sorted(list(chars))

        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        # cut the text in semi-redundant sequences of sequence_length characters
        sentences = []
        next_chars = []

        for text in texts:
            for i in range(0, len(text) - sequence_length, step):
                sentences.append(text[i: i + sequence_length])
                next_chars.append(text[i + sequence_length])

        return chars, char_indices, indices_char, sentences, next_chars
    ## create_sentences ##


    def vectorize(self, sentences, sequence_length, chars, next_chars, char_indices):
        print('\nVectorization...')
        x = np.zeros((len(sentences), sequence_length, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        print ('x.shape', x.shape)
        print ('y.shape', x.shape)

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                j = char_indices[char]
                x[i, t, j] = 1
            y[i, char_indices[next_chars[i]]] = 1

        return x, y
    ## vectorize ##
    """

    def vectorize(self, texts, sequence_length, step):
        print('\nVectorization...')
        # find out characters and create (reverse) character indices
        chars = set()
        for text in texts:
            chars = chars.union(set(text))

        chars = sorted(list(chars))

        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        # cut the text in semi-redundant sequences of sequence_length characters
        sentences = []
        next_chars = []

        for text in texts:
            for i in range(0, len(text) - sequence_length, step):
                sentences.append(text[i: i + sequence_length])
                next_chars.append(text[i + sequence_length])

        # and finally vectorize

        x = np.zeros((len(sentences), sequence_length, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        print ('x.shape', x.shape)
        print ('y.shape', x.shape)

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                j = char_indices[char]
                x[i, t, j] = 1
            y[i, char_indices[next_chars[i]]] = 1

        return x, y, chars, char_indices, indices_char
    ## vectorize ##


    def plot_run(self, loss, acc, val_loss, val_acc, title):
        epochs = range(len(loss))

        plt.figure(figsize=(12,5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, acc, 'go', label='Training accuracy')
        plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

        return
    ## plot_run ##


    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probs = np.random.multinomial(1, preds, 1)

        return np.argmax(probs)

    ## sample ##

    def do_epoch_end_text(self, epoch, logs):
        # Function invoked at end of each epoch. Prints generated text.
        text = self.text_data[0]
        start_index = random.randint(0, len(text) - self.sequence_length - 1)
        with open(self.log_path, 'a') as outfile:
            print(file = outfile)
            print('--- Epoch: {:d}, accuracy: {:.2f}, val-accuracy: {:.2f}, loss: {:.2f}, val-loss: {:.2f}'.format(
                    epoch, logs['accuracy'], logs['val_accuracy'], logs['loss'], logs['val_loss']),
                    file = outfile)

            for diversity in [0.2, 0.5, 1.0, 1.2]:
                print('---- diversity:', diversity, file = outfile)

                generated = ''
                sentence = text[start_index: start_index + self.sequence_length]
                print('----- Generating with seed: "' + sentence + '"', file = outfile)
                #print(generated, file = outfile)

                # find random length
                for i in range(n_results):
                    x_pred = np.zeros((1, self.sequence_length, len(self.chars)))
                    for t, char in enumerate(sentence):
                        x_pred[0, t, self.char_indices[char]] = 1.

                    preds = self.model.predict(x_pred, verbose=0)[0]
                    #print(preds, file = outfile)
                    #for diversity in [0.2, 0.5, 1.0, 1.2]:
                        #print('Diversity', diversity, file = outfile)
                    #    for ii in range(10):
                    #        next_index = sample(preds, diversity)
                            #print('** argmax', next_index, file = outfile)
                    #input('Press <Enter>')

                    next_index = self.sample(preds, diversity)
                    next_char = self.indices_char[next_index]

                    generated += next_char
                    sentence = sentence[1:] + next_char
                # for

                print(generated, '\n', file = outfile)
            # for

        return

    ## do_epoch_end_text ##


    def do_epoch_end_names(self, epoch, logs):
        # Function invoked at end of each epoch. Prints generated text.

        with open(self.log_path, 'a') as outfile:
            print(file = outfile)
            print('--- Epoch: {:d}, accuracy: {:.2f}, val-accuracy: {:.2f}, loss: {:.2f}, val-loss: {:.2f}'.format(
                    epoch, logs['accuracy'], logs['val_accuracy'], logs['loss'], logs['val_loss']),
                    file = outfile)

            for diversity in [0.2, 0.5, 1.0, 1.2]:
                # outfile, texts, sequence_length, prefixes, diversity, n
                self.create_examples(self.model, outfile, self.text_data, self.sequence_length,
                                    self.prefixes, diversity, n_results)

        return

    ## do_epoch_end_names ##


    def create_model (self, char_set, l_size, dropout, sequence_length, bi_dir):
        # build the model: a single RNN layer

        print('Build model...')

        model = Sequential()
        if bi_dir:
            model.add(Bidirectional (layer_type(l_size,
                                                input_shape=(sequence_length, len(char_set))
                                               )
                                    )
                     )
                                 #kernel_regularizer=regularizers.l2(0.01),
                                 #activity_regularizer=regularizers.l2(0.01)
                                 #return_sequences=True))


            model.add(Dropout(dropout))

            #model.add(layer_type(l_size))
            #model.add(Dropout(dropout))

            model.add(Dense(len(char_set), activation='relu'))
            model.add(Activation('softmax'))
        else:
            model.add(layer_type(int(l_size / 2),
                                 dropout = dropout,
                                 recurrent_dropout = dropout,
                                 input_shape = (sequence_length, len(char_set)),
                                 return_sequences = True))
            model.add(layer_type(l_size,
                                 dropout = dropout,
                                 recurrent_dropout = dropout
                                 ))

            model.add(Dense(len(char_set), activation='relu'))
            model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01) # Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

        #print(model.summary())

        return model
    ## create_model ##


    def create_examples(self, model, outfile, texts, prefixes,
                        chars, char_indices, indices_char,
                        sequence_length, diversity, n):
        if not outfile is None:
            print('---- diversity:', diversity, file = outfile)

        generates = []

        # For each result, find a name and generate a new name
        for i in range(n):
            generated = ''
            index = random.randint(0, len(texts) - 1)
            text = texts[index]
            start_index = random.randint(0, len(text) - sequence_length - 1)

            sentence = text[start_index: start_index + sequence_length]
            if not outfile is None:
                outfile.write('----- Generating with seed: "' + sentence + '": ')

            end_of_name = False
            i = 0
            while not end_of_name:
                x_pred = np.zeros((1, sequence_length, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                i += 1
                end_of_name = i > 32 #next_char == '.'
            # while

            generates.append(generated)
            if not outfile is None:
                outfile.write(generated + '\n')

        # for
        return generates
    ## create_examples ##


    def post_process(self, names, pref_x, known_names):
        new_names = []
        for name in names:
            namen = name.split(' ')
            prefixes = list(pref_x.keys())
            weights = list(pref_x.values())


            parts = [word for word in namen if not word in prefixes]
            namen = [word for word in parts if len(word) > 1]
            while len(namen) > 1:
                prefix = random.choices(prefixes, weights)
                prefix = ''.join(prefix)
                voornaam = namen[0]
                achternaam = namen [1]
                if len(prefix) > 0:
                    naam = voornaam + ' ' + prefix + ' ' + achternaam
                else:
                    naam = voornaam + ' ' + achternaam

                if not naam in known_names:
                    new_names.append(naam)

                namen = namen[2:]
            # while
        # for

        return new_names


    def run_task(self, text_data, prefixes, sequence_length, step, epochs, layer_size,
                 dropout, bidirectional):

        # Forward declarations for do_epoch_end
        model = None
        chars = None
        char_indices = None
        indices_char = None

        def do_epoch_end(epoch, logs):
            # Function invoked at end of each epoch. Prints generated text.

            with open(self.log_path, 'a') as outfile:
                print(file = outfile)
                print('--- Epoch: {:d}, accuracy: {:.2f}, val-accuracy: {:.2f}, loss: {:.2f}, val-loss: {:.2f}'.format(
                        epoch, logs['accuracy'], logs['val_accuracy'], logs['loss'], logs['val_loss']),
                        file = outfile)

                for diversity in [0.2, 0.5, 1.0, 1.2]:
                    # outfile, texts, sequence_length, prefixes, diversity, n
                    self.create_examples(model, outfile, text_data, prefixes,
                                         chars, char_indices, indices_char,
                                         sequence_length, diversity, n_results)

            return

        ## do_epoch_end_names ##

        seconds = time.time()
        x, y, chars, char_indices, indices_char = self.vectorize(
            text_data, sequence_length, step)

        #plot_learning = PlotProgress(epochs, 5, chars)
        if self.data_source == 'names':
            print_callback = LambdaCallback(on_epoch_end=do_epoch_end)
        else:
            print_callback = LambdaCallback(on_epoch_end=self.do_epoch_end_text)

        early_stop = EarlyStopping(monitor='val_accuracy',
                                   min_delta=0.001,
                                   patience=5,
                                   verbose=1,
                                   mode='auto')

        reduce_lr = ReduceLROnPlateau(
                monitor='loss',
                factor = 0.5,
                patience = 1,
                verbose = 1,
                min_lr = 1e-05)

        # Create model
        model = self.create_model(chars, layer_size, dropout,
                                       sequence_length, bidirectional)

        print(model.summary())

        # Show model
        with open(self.log_path, 'a') as outfile:
            info = ('\n' + 80 * '=' + '\n' +
                    '*** Data source      ' + str(self.data_source) + '\n' +
                    '*** Layer size:      ' + str(layer_size) + '\n' +
                    '*** Bidirectional:   ' + str(bidirectional) + '\n' +
                    '*** Dropout:         ' + str(dropout) + '\n' +
                    '*** # sequences:     ' + str(x.shape[0]) + '\n' +
                    '*** Sequence length: ' + str(sequence_length) + '\n\n')
            sys.stdout.write(info)
            outfile.write(info)

            # write summary to file, use print_fn param with lambda
            #model.summary(print_fn=lambda x: outfile.write(x + '\n'))
            outfile.flush()

            self.sequence_length = sequence_length
            hist = model.fit(x, y,
                      batch_size=128,
                      epochs=epochs,
                      validation_split=0.25,
                      callbacks=[reduce_lr, print_callback, early_stop])

            # Save the model
            pad = self.model_path.format(layer_size, int(100*dropout), sequence_length)
            model.save(pad)

            # plot the result of the training
            hist = model.history
            self.plot_run(hist.history['loss'], hist.history['accuracy'],
                     hist.history['val_loss'], hist.history['val_accuracy'],
                     title='Character generation for ' + str(epochs) \
                           + ' epochs, layer size: ' + str(layer_size) + ', dropout: '\
                           + str(dropout) + ', seq length: ' + str(sequence_length)
                    )

            # Fetch history, print message and return model and history
            loss = hist.history['loss']
            acc = hist.history['accuracy']
            val_loss = hist.history['val_loss']
            val_acc = hist.history['val_accuracy']
            ma = np.argmax(val_acc)
            ml = np.argmin(val_loss)
            seconds = time.time() - seconds
            cpu_text = '\n\nTotal CPU time of this run was ' + str(int(seconds)) + ' seconds.\n'
            print(cpu_text)
            outfile.write(cpu_text)
            outfile.write('Maximum validation accuracy at epoch {:d} is {:6.4f}\n'.format(ma, val_acc[ma]))
            outfile.write('Minimum validation loss at epoch {:d} is {:6.4f}\n'.format(ml, val_loss[ml]))

        return model, chars, char_indices, indices_char, acc[-1], val_acc[-1], loss[-1], val_loss[-1], seconds
    ## run_task ##


    def action_train(self, epochs, layer_sizes, dropouts,
                      sequence_lengths, step, bidirectional, examples):

        # Read or create the data
        text_data, prefixes = sequencer.get_data(source, seq_list)

        # Run the model with the specified parameters
        with open(self.summ_path, 'w', buffering = 1) as summ_file:
            print('Epochs Bi Size Dropout Seqlen  Acc ValAcc    Loss ValLoss    CPU', file = summ_file)
            for layer_size in layer_sizes:
                for dropout in dropouts:
                    for sequence_length in sequence_lengths:
                        # Create and train a model with current layer_size, dropout and sequence length
                        model, chars, char_indices, indices_char, acc, val_acc, loss, val_loss, cpu = \
                            self.run_task(text_data, prefixes, sequence_length, step, n_epochs,
                                          layer_size, dropout, bidirectional)

                        # Print results of the training to the summary file
                        print('{:6d} {:2d} {:4d} {:7.2f} {:6d} {:4.2f} {:6.2f} {:7.4f} {:7.4f} {:6d}'.format
                              (epochs, int(bidirectional), layer_size, dropout,
                               sequence_length, acc, val_acc, loss, val_loss, int(cpu + 0.5)),
                              file = summ_file)

                        # when names, examples sample names
                        if self.data_source == 'names' and not examples is None:
                            label = '\n=== Results for size: {:d}, dropout: {:4.2f}, ' + \
                                    'sequence length {:d} ===' \
                                    .format(layer_size, dropout, sequence_length)

                            examples['report_path'] = self.report_path
                            self.generate_names(model, label, text_data, prefixes,
                                                chars, char_indices, indices_char,
                                                sequence_length, self.report_path, examples)
                    # for
                # for
            # for
        # with

        # Ready with all trainings, show a summary of all activity on stdout
        print()
        with open(self.summ_path, 'r') as summ_file:
            for line in summ_file:
                print(line, end = '')

    ## action_train ##


    def write_csv(self, namen, names_name):

        split_names = [x.split() for x in namen]
        voornamen = [x[0] for x in split_names]
        achternamen = [x[-1] for x in split_names]
        prefixes = [x[1:-1] for x in split_names]

        df = pd.DataFrame([[x,y,z] for x,y,z in zip(voornamen, prefixes, achternamen)],
                          columns=['Voornaam', 'Prefixes', 'Achternaam'])

        with open(names_name, 'w') as csv_file:
            print('Voornaam;Prefixes;Achternaam', file=csv_file)
            for x, y, z in zip(voornamen, prefixes, achternamen):
                try:
                    y = y[0]
                except:
                    y = ''

                csv_file.write(x + ';' + y + ';' + z + ';\n')

        return df


    def generate_names(self, model, label, known_names, prefixes,
                       chars, char_indices, indices_char,
                       sequence_length, report_path, generate):

        print('Generating names...')
        diversities = generate['diversities']
        n_names = generate['n']

        seconds = time.time()
        names_generated = 0
        print('*** Sequence length', sequence_length)

        all_names = []
        with open(report_path, 'a') as outfile:
            print(label, file = outfile)

            for diversity in diversities:
                #print('\nDiversity = ' + str(diversity), file = outfile)

                if self.data_source == 'names':
                    names = self.create_examples(model, None, known_names, prefixes,
                                                 chars, char_indices, indices_char,
                                                 sequence_length, diversity, n_names)
                    names = self.post_process(names, prefixes, known_names)
                    names_generated += len(names)

                    for name in names:
                        all_names.append(name)
                # if
            # for

            for name in all_names:
                print(name, file = outfile)
        # with

        seconds = time.time() - seconds
        cpu_text = '\n\nGenerating ' + str(int(names_generated)) + ' names took ' + \
            str(int(seconds)) + ' CPU seconds\n'
        print(cpu_text)

        return all_names
    ## generate_names ##

    def action_generate(self, generate):
        model_path = generate['model_name']
        report_path = self.template.format(self.count, generate['names_name']) + '.txt'
        csv_path = self.template.format(self.count, generate['names_name']) + '.csv'

        model = load_model(model_path)

        sequence_length = model._layers[0].batch_input_shape[1]
        seq_list = [sequence_length]

        mesg = 'Model loaded from: ' + str(model_path) + ', sequence length = ' + \
            str(sequence_length) + ', step size = ' + str(step_size)
        print(mesg)

        # Read or create the data
        text_data, prefixes = self.get_data(source, seq_list)

        _, _, chars, char_indices, indices_char = sequencer.vectorize(
            text_data, sequence_length, step_size)

        info = ('\n' + 80 * '=' + '\n' +
                '*** Data source      ' + str(self.data_source) + '\n' +
                '*** Model from:      ' + model_path + '\n' +
                '*** Names to:        ' + report_path + '\n' +
                '*** Sequence length: ' + str(sequence_length) + '\n' +
                '*** Diversities:     ' + str(generate['diversities']) + '\n' +
                '*** Names:           ' + str(generate['n']) + '\n\n')

        print(info)

        names = self.generate_names(model, mesg, text_data, prefixes,
                            chars, char_indices, indices_char,
                            sequence_length, report_path, generate)

        self.write_csv(names, csv_path)

        return names

    ## action_generate ##


## Class: Sequencing ##


layer_type = GRU#CuDNNGRU
hyper_file = 'create-names.yaml' # Configuration file

if __name__ == '__main__':
    with open(hyper_file) as yaml_data:
        hyper_pars = yaml.load(yaml_data, Loader=yaml.FullLoader)

    ## Data source selection
    config = hyper_pars['names']

    # Retrieve and project parameters
    action = config['action']
    source = config['data']
    step_size = config['step']
    n_epochs = config['epochs']
    layer_list = config['layer_sizes']
    dropout_list = config['dropouts']
    seq_list = config['maxlens']
    bidirectional = config['bidirectional']
    save_path = config['snapshot_path']
    n_samples = config['n_samples']
    n_results = config['n_results']

    generate = config['generate'] if 'generate' in config else None
    examples = config['examples'] if 'examples' in config else None

    # List the parameters used by this model
    print('',
          'Data source:   ', source, '\n',
          'Step size:     ', step_size, '\n',
          'Epochs:        ', n_epochs, '\n',
          'Layer sizes:   ', str(layer_list), '\n',
          'Dropouts:      ', str(dropout_list), '\n',
          'Bidirectional: ', str(bidirectional), '\n',
          'Max lengths:   ', str(seq_list), '\n',
          'N examples:    ', str(n_samples), '\n',
          'N results:     ', str(n_results), '\n')

    # Create sequencing instance
    sequencer = Sequencing(save_path, source)

    if action == 'train':
        # Run the training tasks
        print('Training the models')
        sequencer.action_train(n_epochs, layer_list, dropout_list, seq_list,
                               step_size, bidirectional, examples)

    elif action == 'generate':
        print('Generating names')
        names = sequencer.action_generate(generate)

    else:
        print('*** Unknown action "' + action + '". Only "train" and "generate" are allowed.')

# if