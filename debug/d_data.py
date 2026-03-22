from data.dataset import VLMDataset

ds = VLMDataset(max_samples=100)

print("Dataset size:", len(ds))

sample = ds[0]

print("\nSample keys:", sample.keys())
print("Image type:", type(sample["image"]))
print("Conversation:", sample["conversations"][:2])
