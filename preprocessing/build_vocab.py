from tqdm import tqdm
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import sys
sys.path.append("/home/frisk17/FeatureNet")
from dataloader import IUXRayDataset, CDD_CESMDataset
from vocabulary import Vocabulary
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import pickle

if __name__ == "__main__":
    nltk.download("punkt_tab")
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    dataset = CDD_CESMDataset("data/cdd_cesm_data.csv",
                            transform=transforms.Compose([transforms.Resize(224),
                                                          transforms.CenterCrop(224),
                                                          transforms.ToTensor(),
                                                          normalize]))
    dataloader = DataLoader(dataset=dataset)

    words = []
    for i, (_, _, report) in tqdm(
            enumerate(dataloader), total=len(dataloader)):
        tokens = word_tokenize(report[0])
        words.extend(tokens)

    vocabulary = Vocabulary()
    vocabulary.add_word("<pad>")
    vocabulary.add_word("<start>")
    vocabulary.add_word("<end>")
    vocabulary.add_word("<unk>")

    word_counter = Counter(words)
    print(word_counter)
    for word in words:
        word = word.lower()

        vocabulary.add_word(word)

    with open("data/vocab.pkl", mode="wb") as file:
        pickle.dump(vocabulary, file)
