from langchain_community.document_loaders import MWDumpLoader
import random

loader = MWDumpLoader(
    file_path="YOUR_PATH_HERE",

    encoding="utf8",
    # namespaces = [0,2,3] Optional list to load only specific namespaces. Loads all namespaces by default.
    skip_redirects=True,  # will skip over pages that just redirect to other pages (or not if False)
    stop_on_error=False,  # will skip over pages that cause parsing errors (or not if False)
)
documents=[]
try:
    documents = loader.load()
    print(f"You have {len(documents)} document(s) in your data ")
except Exception as e:
    print(documents)


random.shuffle(documents)
train_data = documents[:int((len(documents)+1)*.90)]
test_data = documents[int((len(documents)+1)*.90):]

f = open("training_full.txt", "w")
for i in train_data:
    f.write(i.page_content)

f = open("validation_full.txt", "w")
for i in test_data:
    f.write(i.page_content)

