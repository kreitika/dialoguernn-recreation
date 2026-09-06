import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class IEMOCAPDataset(Dataset):
    """
    Loads the official IEMOCAP_features.pkl used in the DialogueRNN paper.

    Pickle structure (per the original authors' format):
      videoIDs, videoSpeakers, videoLabels, videoText,
      videoAudio, videoVisual, videoSentence, trainVid, testVid
    We use videoText only (text-only track, matches Table 2 of the paper).
    """

    def __init__(self, path="data/DialogueRNN_features/IEMOCAP_features/IEMOCAP_features_raw.pkl", train=True):
        with open(path, "rb") as f:
            (self.videoIDs, self.videoSpeakers, self.videoLabels,
             self.videoText, self.videoAudio, self.videoVisual,
             self.videoSentence, self.trainVid, self.testVid) = pickle.load(f, encoding="latin1")

        self.keys = list(self.trainVid) if train else list(self.testVid)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]

        text = torch.FloatTensor(np.array(self.videoText[vid]))
        speakers = torch.FloatTensor(
            [[1, 0] if s == "M" else [0, 1] for s in self.videoSpeakers[vid]]
        )                                                       # (num_utterances, 2) one-hot
        labels = torch.LongTensor(self.videoLabels[vid])        # (num_utterances,)
        umask = torch.FloatTensor([1] * len(labels))            # for padding later (all 1s here)

        return text, speakers, labels, umask, vid

    def collate_fn(self, batch):
        """
        Pads variable-length dialogues to the same length within a batch.
        """
        text, speakers, labels, umask, vid = zip(*batch)

        text = torch.nn.utils.rnn.pad_sequence(text)
        speakers = torch.nn.utils.rnn.pad_sequence(speakers)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        umask = torch.nn.utils.rnn.pad_sequence(umask, batch_first=True)

        return text, speakers, labels, umask, vid
