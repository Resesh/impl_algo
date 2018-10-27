#Encoderdecoderを使ったtranslationの実装
#pytorch

import random
import csv
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from nltk import bleu_score
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
try:
    from utils import Vocab
except ModuleNotFoundError:  # iLect環境
    import os
    os.chdir('/root/userspace/chap3/')
    from utils import Vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
random_state = 42

PAD = 0
UNK = 1
BOS = 2
EOS = 3
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
BOS_TOKEN = '<S>'
EOS_TOKEN = '</S>'


def load_data(file_path):
    data = []
    for line in open(file_path, encoding='utf-8'):
        words = line.strip().split()  # スペースで単語を分割
        data.append(words)
    return data


def load_dataset():
    # Load dataset
    train_X = load_data('./data/train.en')
    train_Y = load_data('./data/train.ja')
    test_X = load_data('./data/test.en')
    
    return train_X, train_Y, test_X


def sentence_to_ids(vocab, sentence):
    """
    単語のリストをインデックスのリストに変換する
    :param vocab: Vocabのインスタンス
    :param sentence: list of str
    :return indices: list of int
    """
    ids = [vocab.word2id.get(word, UNK) for word in sentence]
    ids = [BOS] + ids + [EOS]  # </S>トークンを末尾に加える
    return ids


def pad_seq(seq, max_length):
    # 系列(seq)が指定の文長(max_length)になるように末尾をパディングする
    res = seq + [PAD for i in range(max_length - len(seq))]
    return res    


class DataLoader(object):
    def __init__(self, X, Y, batch_size, shuffle = False):
        """
        X:入力単語の文章(単語IDのリスト)のリスト
        Y:出力単語の文章(単語IDのリスト)のリスト
        """
        self.data = list(zip(X, Y))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_index = 0
        
        self.reset()
    
    def __iter__(self):
        return self
    
    def reset(self):
        if self.shuffle:
            self.data = shuffle(self.data, random_state = random_state)
            
        self.start_index = 0
            
    def __next__(self):
        if self.start_index >= len(self.data):
            self.reset()
            raise StopIteration
        
        seqs_X, seqs_Y = zip(*self.data[self.start_index:self.start_index + self.batch_size])
        seq_pairs = sorted(zip(seqs_X, seqs_Y), key = lambda p:len(p[0]), reverse = True)
        seqs_X, seqs_Y = zip(*seq_pairs)
        
        lengths_X = [len(s) for s in seqs_X]
        lengths_Y = [len(s) for s in seqs_Y]
        max_length_X = max(lengths_X)
        max_length_Y = max(lengths_Y)
        padded_X = [pad_seq(s, max_length_X) for s in seqs_X]
        padded_Y = [pad_seq(s, max_length_Y) for s in seqs_Y]
        
        #convert to tensor
        batch_X = torch.tensor(padded_X, dtype = torch.long, device = device).transpose(0,1)
        batch_Y = torch.tensor(padded_Y, dtype = torch.long, device = device).transpose(0,1)
        
        #renew_pointer
        self.start_index += self.batch_size
        
        return batch_X, batch_Y, lengths_X
        
        
        
        

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        input_size:入力言語の語彙数
        hidden_size:隠れ層のユニット数(今回は埋め込みの次元でもある)
        """
        
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx = PAD)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
    def forward(self, seqs, input_lengths, hidden = None):
        """
        seqs:入力のバッチ, size = (max_length, batch_size)
        input_lengths:入力のバッチの各サンプルの文の長さ
        hidden:隠れ層の初期値でNoneなら0
        return hidden: Encoderの隠れ状態, size = (1, batch_size, hidden_size)
        """
        
        emb = self.embedding(seqs)
        #packedsequence型に変換
        packed = pack_padded_sequence(emb, input_lengths)
        #RNNsで出力
        output, hidden = self.gru(packed, hidden)
        #RNNsでの入力を元のtensorに戻す
        output, _ = pad_packed_sequence(output)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, seqs, hidden):
        emb = self.embedding(seqs)
        output, hidden = self.gru(emb, hidden)
        output = self.out(output)
        return output, hidden


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)
        
    def forward(self, batch_X, lengths_X, max_length, batch_Y = None, use_teacher_forcing = False):
        """
        batch_X:入力系列のバッチ, size = (max_length, batch_size)
        lengths_X: 入力系列の各サンプルの文の長さ
        max_length: Decoderの最大の文の長さ
        batch_Y:ターゲット系列
        decoder_output: Decoderの出力tensor, size = (max_length, batch_size, self.decoder.output_size)
        """
        
        #encoderに系列を入力
        _, encoder_hidden = self.encoder(batch_X, lengths_X)
        
        _batch_size = batch_X.size(1)
        
        #decoderの入力と隠れ層の状態定義
        decoder_input = torch.tensor([BOS] * _batch_size, dtype = torch.long, device = device)
        decoder_input = decoder_input.unsqueeze(0)
        decoder_hidden = encoder_hidden #encoderの最終隠れ状態
        
        #decoderの出力のホルダーを定義
        #後々の [t] に代入するため
        
        decoder_outputs = torch.zeros(max_length, _batch_size, self.decoder.output_size, device = device)
        
        for i in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs[t] = decoder_output
            
            if use_teacher_forcing and batch_Y is not None:
                decoder_input = batch_Y[t].unsqueeze(0)
            else:
                decoder_input = decoder_output.max(-1)[1]
                
        return decoder_outputs


mce = nn.CrossEntropyLoss(size_average=False, ignore_index=PAD)
def masked_cross_entropy(logits, target):
    return mce(logits.view(-1, logits.size(-1)), target.view(-1))        


def compute_loss(batch_X, batch_Y, lengths_X, model, optimizer=None, is_train=True):
    # 損失を計算する関数
    model.train(is_train)  # train/evalモードの切替え
    
    # 一定確率でTeacher Forcingを行う
    use_teacher_forcing = is_train and (random.random() < teacher_forcing_rate)
    max_length = batch_Y.size(0)
    # 推論
    pred_Y = model(batch_X, lengths_X, max_length, batch_Y, use_teacher_forcing)
    
    # 損失関数を計算
    loss = masked_cross_entropy(pred_Y.contiguous(), batch_Y.contiguous())
    
    if is_train:  # 訓練時はパラメータを更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    batch_Y = batch_Y.transpose(0, 1).contiguous().data.cpu().tolist()
    pred = pred_Y.max(dim=-1)[1].data.cpu().numpy().T.tolist()

    return loss.item(), batch_Y, pred


def calc_bleu(refs, hyps):
    refs = [[ref[:ref.index(EOS)]] for ref in refs]
    hyps = [hyp[:hyp.index(EOS)] if EOS in hyp else hyp for hyp in hyps]
    return 100 * bleu_score.corpus_bleu(refs, hyps)


# ハイパーパラメータ
min_count = 2 # WRITE ME!
hidden_size = 256 # WRITE ME!
batch_size = 64 # WRITE ME!
num_epochs = 30 # WRITE ME!
lr = 1e-3 # WRITE ME!
teacher_forcing_rate = 0.2 # WRITE ME!
test_max_length = 20 # WRITE ME!


train_X, train_Y, test_X = load_dataset()

train_X, valid_X, train_Y, valid_Y = train_test_split(
    train_X, train_Y, test_size=0.1, random_state=42)


word2id = {
    PAD_TOKEN: PAD,
    BOS_TOKEN: BOS,
    EOS_TOKEN: EOS,
    UNK_TOKEN: UNK,
    }

vocab_X = Vocab(word2id=word2id)
vocab_Y = Vocab(word2id=word2id)
vocab_X.build_vocab(train_X, min_count=min_count)
vocab_Y.build_vocab(train_Y, min_count=min_count)

vocab_size_X = len(vocab_X.id2word)
vocab_size_Y = len(vocab_Y.id2word)

train_X = [sentence_to_ids(vocab_X, sentence) for sentence in train_X]
train_Y = [sentence_to_ids(vocab_Y, sentence) for sentence in train_Y]
valid_X = [sentence_to_ids(vocab_X, sentence) for sentence in valid_X]
valid_Y = [sentence_to_ids(vocab_Y, sentence) for sentence in valid_Y]

train_dataloader = DataLoader(train_X, train_Y, batch_size)
valid_dataloader = DataLoader(valid_X, valid_Y, batch_size, shuffle=False)

model_args = {
    'input_size': vocab_size_X,
    'output_size': vocab_size_Y,
    'hidden_size': hidden_size,
}

model = EncoderDecoder(**model_args).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# 訓練
best_valid_bleu = 0.

for epoch in range(1, num_epochs+1):
    train_loss = 0.
    train_refs = []
    train_hyps = []
    valid_loss = 0.
    valid_refs = []
    valid_hyps = []
    # train
    for batch in train_dataloader:
        batch_X, batch_Y, lengths_X = batch
        loss, gold, pred = compute_loss(
            batch_X, batch_Y, lengths_X, model, optimizer, 
            is_train=True
            )
        train_loss += loss
        train_refs += gold
        train_hyps += pred
    # valid
    for batch in valid_dataloader:
        batch_X, batch_Y, lengths_X = batch
        loss, gold, pred = compute_loss(
            batch_X, batch_Y, lengths_X, model, 
            is_train=False
            )
        valid_loss += loss
        valid_refs += gold
        valid_hyps += pred
        
        
    
    # 損失をサンプル数で割って正規化
    train_loss /= len(train_dataloader.data) 
    valid_loss /= len(valid_dataloader.data) 
    # BLEUを計算
    train_bleu = calc_bleu(train_refs, train_hyps)
    valid_bleu = calc_bleu(valid_refs, valid_hyps)

    # validationデータでBLEUが改善した場合にはモデルを保存
    if valid_bleu > best_valid_bleu:
        ckpt = model.state_dict()
        best_valid_bleu = valid_bleu

    print('Epoch {}: train_loss: {:5.2f}  train_bleu: {:2.2f}  valid_loss: {:5.2f}  valid_bleu: {:2.2f}'.format(
            epoch, train_loss, train_bleu, valid_loss, valid_bleu))
    print('-'*80)


# 学習済みモデルで生成
model.load_state_dict(ckpt)  # 最良のモデルを読み込み
model.eval()

test_X = [sentence_to_ids(vocab_X, sentence) for sentence in test_X]
test_dataloader = DataLoader(test_X, test_X, 1, shuffle=False)  # 演習のDataLoaderをそのまま使う場合はYにダミーとしてtest_Xを与える

pred_Y = []
for batch in test_dataloader:
    batch_X, _, lengths_X = batch
    pred = model(batch_X, lengths_X, max_length=test_max_length)
    pred = pred.max(dim=-1)[1].view(-1).data.cpu().numpy().tolist()
    if EOS in pred:
        pred = pred[:pred.index(EOS)]
    pred_y = [vocab_Y.id2word[_id] for _id in pred] 
    pred_Y.append(pred_y)


with open('submission.csv', 'w') as f:
    writer = csv.writer(f, delimiter=' ', lineterminator='\n')
    writer.writerows(pred_Y)