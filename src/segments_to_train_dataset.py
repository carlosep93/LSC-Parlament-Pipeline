from argparse import ArgumentParser
from pathlib import Path
from json import load
import pandas as pd
import tqdm
import random
from collections import Counter
import json
import numpy as np


def split_sessions(session_ids, seed=42):
    random.Random(seed).shuffle(session_ids)

    total = len(session_ids)
    train_end = int(0.9 * total)
    dev_end = train_end + int(np.ceil(0.05 * total))

    train = session_ids[:train_end]
    dev = session_ids[train_end:dev_end]
    test = session_ids[dev_end:]

    return train, dev, test


def split_df(df, train_frac=0.6, max_dev_size=2000, max_test_size=2000, random_state=42):
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    N = len(df)

    dev_and_test_frac = 1 - train_frac

    dev_size = min(int(dev_and_test_frac / 2 * N), max_dev_size)
    test_size = min(int(dev_and_test_frac / 2 * N), max_test_size)
    train_size = N - dev_size - test_size

    train_df = df.iloc[:train_size].reset_index(drop=True)
    dev_df = df.iloc[train_size:train_size + dev_size].reset_index(drop=True)
    test_df = df.iloc[train_size + dev_size:].reset_index(drop=True)

    return train_df, dev_df, test_df


def get_data(sessions, max_cer_score, in_video_dir, languages):
    data = []
    columns = ["signal", "signal_start", "signal_end", "encoder_prompt", "decoder_prompt", "output", "output_cased",
               "audio_start", "audio_end"]
    for session_id in tqdm.tqdm(sessions):
        for speaker_file in session_id.iterdir():
            speaker_info = load(open(speaker_file))
            for segment in speaker_info["results"][0]:
                if segment["cer_score"] < max_cer_score and languages and segment["language"] in languages:
                    video_file = in_video_dir.joinpath(f"{session_id.stem}/{segment['block']}.mp4")
                    if not video_file.exists(): continue
                    data.append(
                        [str(video_file), segment["sign_start"] * 1000, segment["sign_end"] * 1000, "__lsc__",
                         f"__{segment['language']}__", segment["org_text"], segment["cased_text"],
                         segment["audio_start"] * 1000, segment["audio_end"] * 1000])
    return pd.DataFrame(data, columns=columns)


def compute_vocab(df):
    vocab = Counter()

    # get audio files
    for idx, row in df.iterrows():
        vocab.update(row.output.split(" "))

    return vocab


def minimize_unknowns_in_test(train_df, test_df):
    train_vocab = compute_vocab(train_df)

    filtered_test_df = test_df[
        test_df["output"].apply(lambda s: all(word in train_vocab for word in s.split()))
    ].reset_index(drop=True)

    return filtered_test_df


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--in_dir", help="Input directory to extract metadata source files", required=True
    )
    parser.add_argument(
        "-s", "--in_dir_video", help="Input directory to get segment source files", required=True
    )
    parser.add_argument(
        "-o", "--out_dir", help="Output directory to store converted files", required=True
    )
    parser.add_argument(
        "-c", "--max_cer_score", help="Maximum CER score to use segment", required=True, type=float,
    )
    parser.add_argument(
        "-l", "--languages", help="Source languages", default=None, nargs='+', type=str
    )
    parser.add_argument(
        "--split_by_sessions", help="Whether to split the datasets by session or each split gets samples "
                                    "from all sessions", action='store_true'
    )
    parser.add_argument(
        "--minimize_unknowns", help="Remove samples from test than include words not seen in "
                                    "the train set", action='store_true'
    )
    parser.add_argument(
        "--random_seed", default=42, type=int
    )
    parser.add_argument(
        "--cased", help="Whether to use cased or uncased text", action='store_true', default=False,
    )
    parser.add_argument(
        "--filter_sessions", help="Whether to filter unwanted sessions or not", action='store_true', default=False,
    )

    args = parser.parse_args()

    if args.split_by_sessions and args.minimize_unknowns:
        raise ValueError("--split_by_sessions and --minimize_unknowns cannot be used together")

    in_metadata_dir = Path(args.in_dir)
    in_video_dir = Path(args.in_dir_video)
    out_dir = Path(args.out_dir).joinpath("-".join(args.languages))
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = list(in_metadata_dir.iterdir())

    if args.filter_sessions:
        sessions_to_exclude = {'80329796', '80330290', '80329795', '13243804', '80328911', '80328235', '13309420'}
        sessions = [session for session in sessions if
                    session.stem not in sessions_to_exclude]

    if args.split_by_sessions:
        sessions_train, sessions_dev, sessions_test = split_sessions(sessions)

        print("train...")
        train_df = get_data(sessions_train, args.max_cer_score, in_video_dir, args.languages)
        print("val...")
        val_df = get_data(sessions_dev, args.max_cer_score, in_video_dir, args.languages)
        print("test...")
        test_df = get_data(sessions_test, args.max_cer_score, in_video_dir, args.languages)

        # concatenate and track origin
        all_dfs = pd.concat(
            [train_df, val_df, test_df],
            keys=['train', 'val', 'test']
        )

        # drop duplicates globally, keep first
        all_dfs_unique = all_dfs[~all_dfs.duplicated(subset='output')]

        # split back into separate dataframes
        train_df_clean = all_dfs_unique.xs('train')
        val_df_clean = all_dfs_unique.xs('val')
        test_df_clean = all_dfs_unique.xs('test')

        train_df_clean.to_csv(f"{out_dir}/train.tsv",
                              sep="\t", index=False,
                              lineterminator="\n")

        val_df_clean.to_csv(f"{out_dir}/validation.tsv",
                            sep="\t",
                            index=False, lineterminator="\n")

        test_df_clean.to_csv(f"{out_dir}/test.tsv", sep="\t",
                             index=False,
                             lineterminator="\n")
    else:
        df = get_data(sessions, args.max_cer_score, in_video_dir, args.languages)

        print(len(df))

        duplicates = df[df.duplicated(subset="output_cased")]
        df = df.drop_duplicates(subset="output_cased")

        if args.cased:
            df["output"] = df.pop("output_cased")
        else:
            df.drop(columns=["output_cased"], inplace=True)

        df.to_csv(f"{out_dir}/total.tsv", sep="\t", index=False, lineterminator="\n")

        train_df, dev_df, test_df = split_df(df)

        if args.minimize_unknowns:
            test_df = minimize_unknowns_in_test(train_df, test_df)

        train_df.to_csv(f"{out_dir}/train.tsv", sep="\t", index=False, lineterminator="\n")
        dev_df.to_csv(f"{out_dir}/validation.tsv", sep="\t", index=False, lineterminator="\n")
        test_df.to_csv(f"{out_dir}/test.tsv", sep="\t", index=False, lineterminator="\n")
        duplicates.to_csv(f"{out_dir}/duplicates.tsv", sep="\t", index=False, lineterminator="\n")

    with open(f"{out_dir}/args.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
