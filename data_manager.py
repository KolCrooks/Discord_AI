import tensorflow as tf
import csv
import re

# https://www3.nd.edu/~busiforc/handouts/cryptography/cryptography%20hints.html
common_digraphs = "th er on an re he in ed nd ha at en es of or nt ea ti to it st io le is ou ar as de rt ve".split(" ")

class DataManager:

    def __init__(self, file, skip_first_line = False, large_vocab = 32, train_split = 0.75) -> None:
        data = []
        self.large_vocab = []
        self.basic_vocab = []
        # Load data
        with open(file, encoding="utf-8") as file:
            reader = csv.reader(file)
            skipped = False
            for line in reader:
                if skip_first_line and not skipped:
                    skipped = True
                    continue
                data.append(line)
        # generate vocab
        histo = {}
        sentences = []
        temp_letter_vocab = {" ": 1000000}
        for [_,line,_] in data:
            if len(line) == 0:
                continue
            line += "\x03"
            sentences.append(line)

            words = line.split(" ")
            for word in words:

                if word == "":
                    continue

                for l in word:
                    if l in temp_letter_vocab:
                        temp_letter_vocab[l] += 1
                    else:
                        temp_letter_vocab[l] = 1

                if len(word) <= 1:
                    continue
                if word in histo:
                    histo[word] += 1
                else:
                    histo[word] = 1
        
        temp_letter_vocab = {k:v for k,v in temp_letter_vocab.items() if v > 10}
        # print(f'Vocab_size: {len(temp_letter_vocab.keys())}')
        self.basic_vocab = list(temp_letter_vocab.keys())
        # self.basic_vocab = [chr(i) for i in range(32,127)]
        sorted_histo = {k: v for k, v in sorted(histo.items(), key=lambda item: -item[1])}
        i = 0
        
        for v in sorted_histo.keys():
            if not i < large_vocab:
                break
            if re.fullmatch(r'\<((\:[A-Za-z(\d)]*\:)|\#|\@\!?)\d*\>', v) is None: 
                self.large_vocab.append(v)
                i += 1

        # Add common digraphs to the set
        for d in common_digraphs:
            if d not in self.large_vocab:
                self.large_vocab.append(d)
        self.large_vocab.sort(key=len, reverse=True)
        
        
        print(f'Large_vocab: {self.large_vocab} \n Basic_Vocab_Len: {len(self.basic_vocab)}')
        self._cached_total_vocab = None
        temp_encoded = []
        prev = None
        for d in sentences:
            e = self.encode(d)
            if len(e) > 1:
                if prev != None:
                    temp_encoded.append((e[:-1], e[1:]))
                prev = e

        temp_encoded = tf.ragged.constant(temp_encoded)

        self.encoded_sentences = tf.data.Dataset.from_tensor_slices(temp_encoded).shuffle(len(data)*2)
        train_size = int(train_split * len(data))
        self.data_train = self.encoded_sentences.take(train_size)
        self.data_test = self.encoded_sentences.skip(train_size)


    def get_vocab(self):
        if self._cached_total_vocab == None:
            temp = self.basic_vocab.copy()
            for v in self.large_vocab:
                temp.append(v)
            self._cached_total_vocab = temp
            return temp
        else:
            return self._cached_total_vocab

    def encode(self, text):
        t = text
        total_vocab = self.get_vocab()
        output = []
        while text != "":
            c = False
            for lv in self.large_vocab:
                if text.startswith(lv):
                    output.append(total_vocab.index(lv))
                    text = text[len(lv):]
                    c = True
                    break
            if c:
                continue
            for sv in self.basic_vocab:
                if text[0] == sv:
                    output.append(total_vocab.index(sv))
                    break

            text = text[1:]

        return output
    
    def decode(self, text_arr):
        total_vocab = self.get_vocab()
        output = ""
        for t in text_arr:
            output += total_vocab[t]
        return output
    
    def sample_train(self, n = 50):

        return self.data_train.shuffle(len(self.data_train)*2).take(n)
        
    def sample_test(self, n = 50):
        return self.data_test.shuffle(len(self.data_test)*2).take(n)
    
    def get_a_starter(self):
        sampled = self.encoded_sentences.shuffle(len(self.encoded_sentences)*2).take(1)
        d = list(sampled.as_numpy_iterator())[0]
        space = self.encode(" ")[0]
        word = []
        for i in d[0]:
            if i == space:
                break
            word.append(i)
        # word.append(space)

        return self.decode(word)
        # return self.decode([d[0][0]])