import configs.config
import data.collator
import data.dataset

config = configs.config.Config()
d = data.dataset.VLMDataset()
c = data.collator.VLMCollator(config)
