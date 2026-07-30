from pathlib import Path


def load_documents(data_folder="data"):
    """
    Load English government document text files.

    Args:
        data_folder (str): Path to the data directory.

    Returns:
        list[dict]: List of documents with text and metadata.
    """

    documents = []

    data_path = Path(data_folder)

    if not data_path.exists():
        print(f"{data_folder} folder not found.")
        return documents

    txt_files = sorted(data_path.rglob("*.pdf.en.txt"))

    for file in txt_files:

        try:

            text = file.read_text(
                encoding="utf-8",
                errors="ignore"
            ).strip()

            if not text:
                continue

            documents.append(
                {
                    "text": text,
                    "source": file.name,
                    "language": "English"
                }
            )

        except Exception as e:

            print(f"Error reading {file.name}: {e}")

    return documents