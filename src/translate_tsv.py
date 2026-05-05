from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from tqdm import tqdm

tqdm.pandas()
from argparse import ArgumentParser


def translate(text):
    message = [{"role": "user", "content": template(text)}]
    date_string = datetime.today().strftime('%Y-%m-%d')

    prompt = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        date_string=date_string
    )

    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    input_length = inputs.shape[1]
    outputs = model.generate(input_ids=inputs.to(model.device),
                             max_new_tokens=400,
                             early_stopping=True,
                             num_beams=5)

    return tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--file_path", required=True
    )
    parser.add_argument(
        "-o", "--out_file_path", required=True
    )

    args = parser.parse_args()

    model_id = "BSC-LT/salamandraTA-7b-instruct"

    source = 'es', 'Spanish'
    target = 'ca', 'Catalan'

    template = lambda x: f"Translate the following text from {source} into {target}.\n{source[1]}: {x} \n{target[1]}:"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    df = pd.read_table(args.file_path)
    df_translated = df.copy()

    df_translated["output"] = df_translated["output"].progress_apply(translate)
    df_translated["decoder_prompt"] = f"__{target[0]}__"

    df_translated.to_csv(args.out_file_path, sep="\t", index=False)
