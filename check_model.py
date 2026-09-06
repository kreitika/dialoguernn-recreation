from model import DialogueRNN
from dataloader import IEMOCAPDataset
from torch.utils.data import DataLoader

train_set = IEMOCAPDataset(train=True)
loader = DataLoader(train_set, batch_size=4, collate_fn=train_set.collate_fn)

model = DialogueRNN()
text, speakers, labels, umask, vid = next(iter(loader))

out = model(text, speakers)
print("Output shape:", out.shape)   # expect (seq_len, batch=4, num_classes=6)