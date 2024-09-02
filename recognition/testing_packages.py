import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import librosa.feature
import os
import tensorflow as tf
from pydub import AudioSegment
from pydub.utils import make_chunks

def mfcc_extraction(input_folder, output_folder, format, chunk_size):
    for j, filename in enumerate(os.listdir(input_folder)):

        audio = AudioSegment.from_file(filename, format=format)

        os.makedirs(output_folder, exist_ok=True)

        chunks = make_chunks(audio, chunk_size)

        for i, chunk in enumerate(chunks):
            if len(chunk) < chunk_size:
                continue

            chunk_name = f"{output_folder}/chunk{i}{j}.wav"
            chunk.export(chunk_name, format="wav")





# sr = 16000

# audio_data, sample_rate = librosa.load("data_test/k1.ogg", sr=sr)
# # # audio_data, sample_rate = librosa.load("data_test/a1.ogg", sr=sr)
# audio_data2, sample_rate2 = librosa.load("data_test/m1.ogg")

# (Лог)-Мел-спектрограмма

# S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=12, fmax=8000)
# S_dB = librosa.power_to_db(S, ref=np.max) #логарифмическая операция
# plt.figure()
# librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=8000)
# plt.colorbar()
# plt.show()

# размер входных данных

# print(S_dB[0][1])
# print(len(S_dB))
# print(len(S_dB[0]))


# fig, ax = plt.subplots()
# ax.plot(S_dB[0])
# plt.show()

# Спектрограмма

# D = librosa.stft(audio_data)
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
#
# plt.figure().set_figwidth(12)
# librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
# plt.colorbar()

# Форма волны

# fig, ax = plt.subplots()
# ax.plot(audio_data2)
# plt.show()

#Частотный спектр

# #Дискретное преобразование Фурье (с исп. окна Хэмминга)
# window = np.hanning(len(audio_data))
# windowed_input = audio_data * window
# dft = np.fft.rfft(windowed_input)
# # Амплитудный спектр
# amplitude = np.abs(dft)
# amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)
# # Частотные столбцы
# frequency = librosa.fft_frequencies(sr=sample_rate, n_fft=len(audio_data))
# plt.figure().set_figwidth(12)
# plt.plot(frequency, amplitude_db)
# plt.xscale('log')
# plt.show()


# def wav_extraction(input_folder, output_folder, audio_format, chunk_size):
#     file_counter = 0
#     if os.path.exists(output_folder) and os.path.isdir(output_folder):
#         with os.scandir(output_folder) as entries:
#             for entry in entries:
#                 if entry.is_file():
#                     file_counter += 1
#
#     for j, filename in enumerate(os.listdir(input_folder)):
#         os.makedirs(output_folder, exist_ok=True)
#         full_filename = os.path.join(input_folder, filename)
#
#         audio = AudioSegment.from_file(full_filename, format=audio_format)
#         chunks = make_chunks(audio, chunk_size)
#
#         for i, chunk in enumerate(chunks):
#             if len(chunk) < chunk_size:
#                 continue
#
#             chunk_name = f"chunk{i}{j + file_counter}.wav"
#             full_chankname = os.path.join(output_folder, chunk_name)
#
#             chunk.export(full_chankname, format="wav")

# sr = 16000
# n_mfcc = 12
# n_mels = 128
#
# audio_data, sampling_rate = librosa.load("chunk00.wav", sr=sr)
# S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=n_mels)
#
# mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S, ref=np.max), sr=sr, n_mfcc=n_mfcc)
#
# mfcc_filename = f"mfcc_chunk.npy"
# np.save(mfcc_filename, mfccs)

# data1 = np.load("my_data_set_mfcc/mfcc_chunk_0.npy")
# print(data1)
# print(data1.shape)
#
# data2 = np.load("my_data_set_mfcc/mfcc_chunk_250.npy")
# print(data2)
# print(data2.shape)
#
#
# data3 = np.load("data_set_mfcc/mfcc_chunk_0.npy")
# print(data3)
# print(data3.shape)


images = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]])
labels = tf.constant([0, 1, 0, 1])

# Создаем датасет, где каждый элемент - кортеж (изображение, метка)
dataset = tf.data.Dataset.from_tensor_slices((images, labels))

for image, label in dataset:
    print("Image:", image.numpy(), "Label:", label.numpy())