import string
import json
from cdifflib import CSequenceMatcher
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import re


def process_json(task):
    dataset_path, out_dataset_path, session, speaker_file = task
    print(f"PID {os.getpid()} processing {session}/{speaker_file}", flush=True)

    in_path = os.path.join(dataset_path, session, speaker_file)
    out_dir = os.path.join(out_dataset_path, session)
    out_path = os.path.join(out_dir, speaker_file)

    os.makedirs(out_dir, exist_ok=True)

    data = read_json_file(in_path)
    data = add_cased_text(data)
    if data:
        write_to_json(data, out_path)
    else:
        print(f"Error with {in_path}")


def normalize_char(c: str):
    """
    Normalize a single character:
    - lowercase
    - remove punctuation
    - keep spaces
    """
    if c in string.punctuation + "«»":
        return ""
    return c.lower()


def build_normalized_text(original_text: str):
    """
    Returns:
      - normalized_text: str
      - index_map: list[int] mapping normalized index -> original index
    """
    normalized_chars = []
    index_map = []
    original_text = re.sub(r'(?<=\w)-(?=\w)', ' ', original_text)
    for i, c in enumerate(original_text):
        nc = normalize_char(c)
        if nc:
            normalized_chars.append(nc)
            index_map.append(i)

    return "".join(normalized_chars), index_map


def normalize_query(text: str):
    return "".join(
        c.lower() for c in text if c not in string.punctuation
    )


def similarity(a: str, b: str) -> float:
    return CSequenceMatcher(None, a, b).ratio()


def find_cased_segment_approx(
        original_text: str,
        uncased_segment: str,
        threshold: float = 0.75,
        window_slop: int = 10,
):
    """
    Finds an approximate match for uncased_segment inside original_text.

    threshold: minimum similarity score (0–1)
    window_slop: how much shorter/longer the window can be vs query
    """
    normalized_text, index_map = build_normalized_text(original_text)
    normalized_query = normalize_query(uncased_segment)

    q_len = len(normalized_query)
    best = None  # (score, start, end)

    min_len = max(1, q_len - window_slop)
    max_len = q_len + window_slop

    for start in range(len(normalized_text)):
        for length in range(min_len, max_len + 1):
            end = start + length
            if end > len(normalized_text):
                break

            window = normalized_text[start:end]
            window = re.sub(r'\s+', ' ', window)
            score = similarity(window, normalized_query)

            if score >= threshold:
                if best is None or score > best[0]:
                    best = (score, start, end)

    if best is None:
        return None

    _, start, end = best

    original_start = index_map[start]
    original_end = index_map[end - 1]

    if original_text[original_end + 1] in string.punctuation + "»":
        original_end = original_end + 1

    return original_text[original_start:original_end + 1]


def find_cased_segment(original_text: str, uncased_segment: str):
    """
    Finds a matching segment of any length in the original text
    and returns it with original casing and punctuation.
    """
    normalized_text, index_map = build_normalized_text(original_text)
    normalized_query = normalize_query(uncased_segment)

    start = normalized_text.find(normalized_query)
    if start == -1:
        return None

    end = start + len(normalized_query) - 1

    original_start = index_map[start]
    original_end = index_map[end]

    if original_text[original_end + 1] in string.punctuation + "»":
        original_end = original_end + 1

    return original_text[original_start:original_end + 1]


def read_json_file(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data


def write_to_json(data, json_path):
    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4, sort_keys=False)


def get_texts_by_speaker(data):
    texts = {}
    for speaker, text in data['block_data']['text']:
        speaker = speaker if not speaker.count(' ') == 2 else ' '.join(speaker.split(' ')[1:]) + ', ' + \
                                                              speaker.split(' ')[0]
        if not speaker in texts:
            texts[speaker] = text
        else:
            texts[speaker] += " " + text

    # from here, only use if we discard "mesa"
    if "mesa" in texts:
        del texts["mesa"]
    # to avoid problems with file 80330959/80330959_final_output_15.json
    new_texts = {data["block_data"]["speaker_meta"]: " ".join(texts.values())}
    texts = new_texts

    return texts


def add_cased_text(data):
    texts = get_texts_by_speaker(data)
    c = 0
    print('Number of results:', len(data['results'][0]))
    for result in data['results'][0]:
        c += 1
        # print('Processing result:', c)
        speaker = result['speaker']
        print(result['org_text'])
        # sometimes there's a disparity between the speaker names, for example 'Garriga Ibáñez, Natàlia' and
        # 'La consellera de Cultura' (both refering to the same person)
        speaker_texts = None
        if speaker not in texts:
            if len(texts.keys()) == 1:
                speaker_texts = list(texts.values())[0]
            else:
                print(f"Something went wrong with {data['block_data']['block_id']}")
        else:
            speaker_texts = texts[speaker]

        if not speaker_texts:
            return None

        cased_text = find_cased_segment(speaker_texts, result['org_text'])

        if cased_text == None:
            cased_text = find_cased_segment_approx(speaker_texts, result['org_text'])
        if cased_text == None:
            print('Warning: Text not found', speaker, result['org_text'])
        else:
            result['cased_text'] = cased_text
        print(cased_text)
    return data


if __name__ == "__main__":
    dataset_path = "final_output/"
    out_dataset_path = "final_output_restored/"

    os.makedirs(out_dataset_path, exist_ok=True)

    tasks = []
    for session in os.listdir(dataset_path):
        session_path = os.path.join(dataset_path, session)
        for speaker_file in os.listdir(session_path):
            tasks.append((dataset_path, out_dataset_path, session, speaker_file))

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print("Num workers:", workers)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(tqdm(
            executor.map(process_json, tasks, chunksize=1),
            total=len(tasks)
        ))
