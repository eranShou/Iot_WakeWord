from IvritAiDataFetch import IvritAiDataFetcher

def main():
    fetcher = IvritAiDataFetcher(base_dir="D:\\ivirit-ai-data\\ivrit-ai dataset")
    words_to_extract = ["שלום"]
    fetcher.extract_word_clips(words_to_extract)

if __name__ == "__main__":
    main()