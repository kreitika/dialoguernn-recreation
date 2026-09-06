from dataloader import IEMOCAPDataset

train_set = IEMOCAPDataset(train=True)
test_set = IEMOCAPDataset(train=False)

print("Train dialogues:", len(train_set))   # paper says 120
print("Test dialogues:", len(test_set))     # paper says 31

text, speakers, labels, umask, vid = train_set[0]
print("Utterance feature shape:", text.shape)   # should be (num_utts, 100)
print("Labels:", labels)