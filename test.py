import ir_datasets

dataset = ir_datasets.load("miracl/id")
i = 0
for doc in dataset.docs_iter():
    print(doc)
    i = i +1
    if i >= 5:
        break