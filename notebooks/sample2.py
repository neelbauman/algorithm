#事前準備
# !pip install janome
# pip install pykakasi

from google.colab import drive
drive.mount('/content/drive')

from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Dense, GRU, Input, Masking
from pykakasi import kakasi
import re
import pickle
import numpy as np

def is_invalid(message):
    is_invalid =False
    for char in message:
        if char not in chars_list:
            is_invalid = True
    return is_invalid

# 文章をone-hot表現に変換する関数
def sentence_to_vector(sentence):
    vector = np.zeros((1, max_length_x, n_char), dtype=np.bool)
    for j, char in enumerate(sentence):
        vector[0][j][char_indices[char]] = 1
    return vector

def respond(message, beta=5):
    vec = sentence_to_vector(message)  # 文字列をone-hot表現に変換
    state_value = encoder_model.predict(vec)
    y_decoder = np.zeros((1, 1, n_char))  # decoderの出力を格納する配列
    y_decoder[0][0][char_indices['\t']] = 1  # decoderの最初の入力はタブ。one-hot表現にする。
    respond_sentence = ""  # 返答の文字列
    while True:
        y, h = decoder_model.predict([y_decoder, state_value])
        p_power = y[0][0] ** beta  # 確率分布の調整
        next_index = np.random.choice(len(p_power), p=p_power/np.sum(p_power))
        next_char = indices_char[next_index]  # 次の文字
        if (next_char == "\n" or len(respond_sentence) >= max_length_x):
            break  # 次の文字が改行のとき、もしくは最大文字数を超えたときは終了
        respond_sentence += next_char
        y_decoder = np.zeros((1, 1, n_char))  # 次の時刻の入力
        y_decoder[0][0][next_index] = 1
        state_value = h  # 次の時刻の状態
    return respond_sentence

#テキストデータの前処理
text = ""
with open("/content/drive/My Drive/ainu_shinyoshu.txt", mode="r", encoding="Shift-JIS") as f:  # ファイルの読み込み
  text_novel = f.read()

text_novel = re.sub("《[^》]+》", "", text_novel)  # ルビの削除
text_novel = re.sub("［[^］]+］", "", text_novel)  # 読みの注意の削除
text_novel = re.sub("〔[^〕]+〕", "", text_novel)  # 読みの注意の削除
text_novel = re.sub("[ 　\n「」『』（）｜※＊…]", "", text_novel)  # 全角半角スペース、改行、その他記号の削除
text += text_novel

print("文字数:", len(text))
print(text)

#漢字をひらがなに変換(kakashi)
seperator = "。"  # 。をセパレータに指定
sentence_list = text.split(seperator)  # セパレーターを使って文章をリストに分割する
sentence_list.pop() # 最後の要素は空の文字列になるので、削除
sentence_list = [x+seperator for x in sentence_list]  # 文章の最後に。を追加
kakasi = kakasi()
kakasi.setMode("J", "H")  # J(漢字) からH(ひらがな)へ
conv = kakasi.getConverter()

for sentence in sentence_list:
    print(sentence)
    print(conv.do(sentence))
    print()

# set()を使って文字の重複を無くし、使用されている文字の一覧を表示
kana_text = conv.do(text)  # 全体をひらがなに変換

print(set(kana_text))  # set()で文字の重複をなくす

#テキストデータの保存
print(kana_text)
with open("kana_ainu.txt", mode="w", encoding="utf-8") as f:
    f.write(kana_text)

#使用する文字
hiragana = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞ\
ただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽ\
まみむめもゃやゅゆょよらりるれろゎわゐゑをん"

katakana = "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾ\
タダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポ\
マミムメモャヤュユョヨラリルレロヮワヰヱヲンヴ"

chars = hiragana + katakana
with open("kana_ainu.txt", mode="r", encoding="utf-8") as f:  # 前回保存したファイル
    text = f.read()

for char in text:  # ひらがな、カタカナ以外でコーパスに使われている文字を追加
    if char not in chars:
        chars += char

chars += "\t\n"  # タブと改行を追加
chars_list = sorted(list(chars))  # 文字列をリストに変換してソートする
print(chars_list)

with open("kana_chars.pickle", mode="wb") as f:  # pickleで保存
    pickle.dump(chars_list, f)


#ここからチャットボット
#文字の読み込み
with open('kana_chars.pickle', mode='rb') as f:
    chars_list = pickle.load(f)

print(chars_list)


#文章のベクトル化
# インデックスと文字で辞書を作成
char_indices = {}
for i, char in enumerate(chars_list):
    char_indices[char] = i

indices_char = {}
for i, char in enumerate(chars_list):
    indices_char[i] = char

n_char = len(chars_list)
max_length_x = 128

#ここまでチャットボット。下は学習。
#各設定

batch_size = 32
epochs = 1000
n_mid = 256  # 中間層のニューロン数

#学習モデルの構築
encoder_input = Input(shape=(None, n_char))
encoder_mask = Masking(mask_value=0)  # 全ての要素が0であるベクトルの入力は無視する
encoder_masked = encoder_mask(encoder_input)

encoder_lstm = GRU(n_mid, dropout=0.2, recurrent_dropout=0.2, return_state=True)  # dropoutを設定し、ニューロンをランダムに無効にする
encoder_output, encoder_state_h = encoder_lstm(encoder_masked)

decoder_input = Input(shape=(None, n_char))
decoder_mask = Masking(mask_value=0)  # 全ての要素が0であるベクトルの入力は無視する
decoder_masked = decoder_mask(decoder_input)

decoder_lstm = GRU(n_mid, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, return_state=True)  # dropoutを設定
decoder_output, _ = decoder_lstm(decoder_masked, initial_state=encoder_state_h)  # encoderの状態を初期状態にする

decoder_dense = Dense(n_char, activation='softmax')
decoder_output = decoder_dense(decoder_output)

model = Model([encoder_input, decoder_input], decoder_output)
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

print(model.summary())

#学習
# val_lossに改善が見られなくなってから、30エポックで学習は終了
early_stopping = EarlyStopping(monitor="val_loss", patience=30)

# ここが怪しい
history = model.fit(
    x = [x_encoder, x_decoder],
    y = t_decoder,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.1,  # 10%は検証用
    callbacks = [early_stopping],
)

"""
Model.fit(
    x=None,
    y=None,
    batch_size=None,
    epochs=1,
    verbose="auto",
    callbacks=None,
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
)
"""

#予測用モデルの構築
# encoderのモデル
encoder_model = Model(encoder_input, encoder_state_h)

# decoderのモデル
decoder_state_in_h = Input(shape=(n_mid,))
decoder_state_in = [decoder_state_in_h]
decoder_output, decoder_state_h = decoder_lstm(
    decoder_input,
    initial_state = decoder_state_in_h,
)

decoder_output = decoder_dense(decoder_output)
decoder_model = Model(
    [decoder_input] + decoder_state_in,
    [decoder_output, decoder_state_h],
)

# モデルの保存
encoder_model.save('encoder_model.h5')
decoder_model.save('decoder_model.h5')

# 'encoder_model.h5'のあるところまで。下からチャットボット。返答作成用の関数。
encoder_model = load_model('encoder_model.h5')
decoder_model = load_model('decoder_model.h5')



#チャットボット本番
bot_name = "bot"
your_name = input("おなまえをおしえてください。:")

print()
print(bot_name + ": " + "こんにちは、" + your_name + "さん。")

message = ""
while message != "さようなら。":
    while True:
        message = input(your_name + ": ")
        if not is_invalid(message):
            break
        else:
            print(bot_name + ": ひらがなか、カタカナをつかってください。")
    response = respond(message)
    print(bot_name + ": " + response)
