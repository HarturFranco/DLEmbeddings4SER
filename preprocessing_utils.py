#
#    Data processing Util file
#
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from os import listdir, scandir
from os.path import join
import librosa
import librosa.display
import numpy as np
import opensmile
from opensmile import Smile
import torch
from transformers import Wav2Vec2ForCTC
import torchaudio
from hmmlearn import hmm
# Load the pre-trained model
model_name = "facebook/wav2vec2-base"
model = Wav2Vec2ForCTC.from_pretrained(model_name)



#  this function
def load_files(dataset_path):
    wavs = [] # list of wav files path's
    labels = [] # list of the corresponding labels
    tess_dir = './TESS' # dataset path

    # get path for all wave files 
    for aud_dir in sorted(scandir(dataset_path), key=lambda x: x.name): 
        dir_files = [join(aud_dir, f) for f in sorted(listdir(aud_dir.path))]
        wavs += dir_files
        labels += [aud_dir.name[4:].lower()] * len(dir_files)

    return wavs, labels


def extract_features(audio_file, sampling_rate=16000, audio_duration=1, truncation=False, padding=False):
    
    
    # loading data
    data, sample_rate = librosa.load(audio_file, sr=sampling_rate)
    
    
    # applying truncation or padding

    if truncation and len(data) > audio_duration * sampling_rate:
        data = data[:int(audio_duration * sampling_rate)]
    elif padding and len(data) < audio_duration * sampling_rate:

        padding_samples = int(audio_duration * sampling_rate) - len(data)
        data = np.pad(data, (0, padding_samples), mode='constant')
    
    
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally
    
    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally
    
    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data,n_mfcc=13, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally
    
    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    melspec = librosa.feature.melspectrogram(y=data, sr=sample_rate).T
    mel = np.mean(melspec, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result



def openSmileFE(audio_file, sampling_rate=16000, audio_duration=1, truncation=False, padding=False):

    signal, sample_rate = librosa.load(audio_file, sr=sampling_rate)

    smile = Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                  feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    result  = smile.process_signal(
                        signal,
                        sampling_rate
                    )
    print('a')    
    return result




# Define a function to extract embeddings from a single audio file
def extract_embeddings(audio_file_path, pooled=False):
    
    # Load the audio file using torchaudio
    waveform, sample_rate = torchaudio.load(audio_file_path)
    
    # Resample audio to the sample rate expected by the model (16kHz)
    if sample_rate != 16000:
        # print(f"original sr={sample_rate}")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        # print(waveform)
    
    # Extract embeddings
    with torch.no_grad():
        output = model(waveform, output_hidden_states=True)# Extract acoustic features
        # print(output.keys())
        embeddings = output.hidden_states
        last_hidden_state = embeddings[-1]

        if pooled:
            pooled_hidden_state = torch.mean(last_hidden_state, dim=1)
            # Assuming `pooled_hidden_state` is your PyTorch tensor
            pooled_hidden_state_np = pooled_hidden_state.detach().numpy()[0]
            last_hidden_state = pooled_hidden_state_np
            # output = model(embeddings)# Extract acoustic features
            print(f"em len: {len(pooled_hidden_state_np)}")
            # You can also obtain the sequence of embeddings if needed
            # sequence_output = model.encoder(embeddings)
    
    return last_hidden_state

# get list of audio files and return s list of embeddings
def get_encoded_embedding(audio_paths, pooled=False):
    embeddings_list = []
    for audio_path in audio_paths:
        embeddings = extract_embeddings(audio_path, pooled)
        embeddings_list.append(embeddings)
    return embeddings_list


##  Evaluation

def eval_model(y_pred, y_test):# Make predictions on the test data

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")


## Hidden Markov model setup

class HMMTrainer(object):
  def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
    self.model_name = model_name
    self.n_components = n_components
    self.cov_type = cov_type
    self.n_iter = n_iter
    self.models = []
    if self.model_name == 'GaussianHMM':
        self.model = hmm.GaussianHMM(n_components=self.n_components,        covariance_type=self.cov_type,n_iter=self.n_iter)
    else:
        raise TypeError('Invalid model type') 

  def train(self, X):
    print("Training...")
    np.seterr(all='ignore')
    self.models.append(self.model.fit(X))
    print("Trained!")
    # Run the model on input data
  def get_score(self, input_data):
    return self.model.score(input_data)
  

def get_HMMs(X_train, y_train, n_classes=7, n_components=7):
    hmm_models = []

    for i in range(n_classes):
        class_indexes = [x for x in range(len(y_train)) if y_train[x] == i]
        print(f"Class len = {len(class_indexes)}")
        label = i
        X = np.array([])

        for idx in class_indexes:
            # print(f"idx = {idx}")
            if len(X) == 0:
                X = X_train[idx]
            else:
                X = np.append(X, X_train[idx], axis=0)
            # print('X.shape =', X.shape)
        # Train and save HMM model
        hmm_trainer = HMMTrainer(n_components=n_components)
        hmm_trainer.train(X)
        hmm_models.append((hmm_trainer, label))
        hmm_trainer = None
        print(f"trained: {i}")
    
    return hmm_models

def HMM_predict(hmm_models, X_test):
    pred_labels = []
    for em in X_test:
        output_label = None
        max_score = -9999999999999999999
        for item in hmm_models:
            hmm_model, label = item
            score = hmm_model.get_score(em)
            if score > max_score:
                max_score = score
                output_label = label
        pred_labels.append(output_label)
    
    return pred_labels