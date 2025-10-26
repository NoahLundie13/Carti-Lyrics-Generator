import os
import re


def clean_lyrics(text: str) -> str:
    text = text.lower()

    text = re.sub(r"\(([^)]+)\)", lambda m: " <ADLIB> " + m.group(1).strip(), text)

    text = re.sub(r"\[(?!verse|chorus|intro|outro).*?\]", "", text)

    section_map = {
        r"\[intro.*?\]": "<INTRO>",
        r"\[verse.*?\]": "<VERSE>",
        r"\[chorus.*?\]": "<CHORUS>",
        r"\[hook.*?\]": "<CHORUS>",
        r"\[outro.*?\]": "<OUTRO>",
    }
    for pattern, token in section_map.items():
        text = re.sub(pattern, f"\n{token}\n", text)

    text = re.sub(r"[^a-zA-Z0-9'\n\s<>\-]", "", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()

    return text


def tag_repeated_lines(text: str, min_repeats=3) -> str:
    """Automatically tag repeated lines as <CHORUS>."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    seen = {}
    for line in lines:
        seen[line] = seen.get(line, 0) + 1
    tagged = []
    for line in lines:
        if seen[line] >= min_repeats:
            tagged.append("<CHORUS>\n" + line)
        else:
            tagged.append(line)
    return "\n".join(tagged)


def build_dataset(input_folder, output_file):
    all_songs = []

    for filename in os.listdir(input_folder):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
            text = f.read()
            cleaned = clean_lyrics(text)
            cleaned = tag_repeated_lines(cleaned)
            formatted = f"<SONG>\n<VERSE>\n{cleaned}\n"
            all_songs.append(formatted)

    dataset = "\n\n".join(all_songs)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(dataset)

    print(f"Cleaned {len(all_songs)} songs and saved to {output_file}")


if __name__ == "__main__":
    build_dataset("dataset", "carti_dataset.txt")
