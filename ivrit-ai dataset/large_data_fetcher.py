from IvritAiDataFetch import IvritAiDataFetcher

def main():
    fetcher = IvritAiDataFetcher()

    # List all folders in the repository
    folders = fetcher.list_transcripts_root_folders()
    print("Folders in the repository:")
    for folder in folders:  # Take the first 40 folders
        print(folder)

        # Download one file from each folder
        fetcher.download_episodes(path=folder, max_per_day=1000)

    # Extract the words "להתראות" and "שלום"
    words_to_extract = ["להתראות", "שלום"]
    fetcher.extract_word_clips(words_to_extract)

if __name__ == "__main__":
    main()