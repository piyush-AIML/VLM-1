from datasets import load_dataset


split=["ocr_1","ocr_2","ocr_3","ocr_4","ocr_5","ocr_6"]

ds=load_dataset("nvidia/Llama-Nemotron-VLM-Dataset-v1")
print(ds)


