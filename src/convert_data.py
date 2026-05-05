import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

from data_converter import DataConverter
from parlament_data_converter import ParlamentDataConverter

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", help="Type of the source files")
    parser.add_argument("-d", "--dest", help="Type of the destination files")
    parser.add_argument(
        "-i", "--in_dir", help="Input directory to extract source files"
    )
    parser.add_argument(
        "-o", "--out_dir", help="Output directory to store converted files"
    )
    parser.add_argument(
        "--process_xml",
        action=BooleanOptionalAction,
        default=False,
        help="Process XML files",
    )

    args = parser.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    data_processor = ParlamentDataConverter(in_dir, out_dir)

    if args.source == "pdf" and args.dest == "xml":
        data_processor.pdf2xml(process_xml=args.process_xml)
    elif args.source == "xml" and args.dest == "yaml":
        data_processor.xml2yaml()
    elif args.source == "pdf" and args.dest == "yaml":
        data_processor.pdf2yaml(process_xml=args.process_xml)
    elif args.source == "pdf" and args.dest == "json":
        data_processor.pdf2json()
    else:
        logging.error("Operation not supported.")


if __name__ == "__main__":
    main()
