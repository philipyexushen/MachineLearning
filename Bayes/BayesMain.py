from Bayes.DocumentClassifier import *

def main():
    list_word, word_classified_map = load_data_set()
    result_vocab_list = create_vocabList(list_word)
    print(set_words_to_vec(result_vocab_list, list_word[0]))

if __name__ == "__main__":
    main()