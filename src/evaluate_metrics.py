import os
from argparse import ArgumentParser
import re
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def read_file(file_path):
    sources = []
    predictions = []

    for line in open(file_path):
        if re.match(r'^L \[\d+\]\s*', line):
            sources.append(re.sub(r'^L \[\d+\]\s*', '', line))
        elif re.match(r'^P \[\d+\]\s*', line):
            predictions.append(re.sub(r'^P \[\d+\]\s*', '', line))
        else:
            print(f"No match: {line}")

    assert len(sources) == len(predictions)
    return sources, predictions


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--in_file", help="Input file with source an predictions", required=True
    )
    parser.add_argument(
        "-m", "--metrics", help="Metrics to be tested", default=None, nargs='+', type=str,
        choices=["bertscore", "comet", "labse", "blaser"]
    )
    parser.add_argument(
        "-o", "--out_path", help="Folder where to save output file", required=True
    )
    parser.add_argument(
        "-s", "--source_language", help="Language of the source text", default=None, choices=["spa", "cat"]
    )
    parser.add_argument(
        "-t", "--target_language", help="Language of the target text", default=None, choices=["spa", "cat"]
    )

    args = parser.parse_args()

    if "blaser" in args.metrics and (not args.source_language or not args.target_language):
        raise ValueError("BLASER metric requires both source_language and target_language.")

    sources, predictions = read_file(args.in_file)

    df = pd.DataFrame({"source": sources, "prediction": predictions})

    for metric in args.metrics:
        print(f"testing {metric}")
        if metric == "bertscore":
            from bert_score import BERTScorer

            scorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli',
                                lang='ca')  # , rescale_with_baseline=True) # not available for catalan
            P, R, F1 = scorer.score(predictions, sources)

            df["bertscore_P"] = P
            df["bertscore_R"] = R
            df["bertscore_F1"] = F1

        elif metric == "comet":
            # 2026 update: setuptools 82.0 removed pkg_resources, so if you still need it, make sure to install version 81 or lower.
            from comet import download_model, load_from_checkpoint

            # need to login with "hf auth login" to access this model (after accepting their terms and conditions)
            model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl", local_files_only=True)

            # Load the model checkpoint:
            model = load_from_checkpoint(model_path)

            data = [{"src": s, "mt": p} for s, p in zip(sources, predictions)]

            # data = [
            #     {
            #         "src": "The output signal provides constant sync so the display never glitches.",
            #         "mt": "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört."
            #     },
            #     {
            #         "src": "Kroužek ilustrace je určen všem milovníkům umění ve věku od 10 do 15 let.",
            #         "mt": "Кільце ілюстрації призначене для всіх любителів мистецтва у віці від 10 до 15 років."
            #     },
            #     {
            #         "src": "Mandela then became South Africa's first black president after his African National Congress party won the 1994 election.",
            #         "mt": "その後、1994年の選挙でアフリカ国民会議派が勝利し、南アフリカ初の黒人大統領となった。"
            #     }
            # ]
            model_output = model.predict(data, batch_size=8, gpus=1)

            df["cometkiwi"] = model_output.scores

        elif metric == "labse":
            from sentence_transformers import SentenceTransformer, util

            model = SentenceTransformer('sentence-transformers/LaBSE')


            def score_similarity(a, b):
                a_embed = model.encode(a)
                b_embed = model.encode(b)
                sim = util.cos_sim(a_embed, b_embed).cpu().numpy().flatten()[0]
                return sim


            df["labse"] = df.progress_apply(lambda x: score_similarity(x["source"], x["prediction"]), axis=1)

        elif metric == "blaser":
            from sonar.models.blaser.loader import load_blaser_model
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

            t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                                       tokenizer="text_sonar_basic_encoder")

            # blaser_qe = load_blaser_model("blaser_st2st_qe_v2_0").eval()
            blaser_qe = load_blaser_model("blaser_2_0_qe").eval()


            def score_similarity(a, b):
                a_embed = t2vec_model.predict([a], source_lang=f"{args.source_language}_Latn")
                b_embed = t2vec_model.predict([b], source_lang=f"{args.target_language}_Latn")
                score = blaser_qe(src=a_embed, mt=b_embed).item()
                return score


            df["blaser"] = df.progress_apply(lambda x: score_similarity(x["source"], x["prediction"]), axis=1)



        else:
            raise NotImplementedError

    os.makedirs(args.out_path, exist_ok=True)
    df.to_csv(os.path.join(args.out_path, f"evaluation_results_{'-'.join(args.metrics)}.tsv"), sep="\t", index=False)

    with open(os.path.join(args.out_path, f"evaluation_results_{'-'.join(args.metrics)}_summary.tsv"),
              "w") as summary_file:
        for col in df.columns:
            if col not in ["source", "prediction"]:
                summary_file.write(f"{col}_mean: {df[col].mean()}\n")
