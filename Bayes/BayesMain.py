from Bayes.DocumentClassifier import *

def main():
    classifier = DocumentClassifier()
    classifier.load_data_set()
    classifier.train()

    ret = classifier.test(["stupid", "worthless", "dog"])
    print(f"result={ret}")

if __name__ == "__main__":
    main()