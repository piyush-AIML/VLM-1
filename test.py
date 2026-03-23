import data.dataset

print("succesful")


ds = data.dataset.VLMDataset(max_samples=500, seed=42)
print(ds)
