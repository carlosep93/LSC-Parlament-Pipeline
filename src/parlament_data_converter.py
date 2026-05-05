import logging
import os
import re
import shutil
from pathlib import Path

from lxml.etree import XMLSyntaxError
from scraperwiki import pdftoxml

from data_converter import DataConverter
from io_file import IO
from xml_parser import XMLParser


class ParlamentDataConverter(DataConverter):
    def __init__(self, in_dir: Path, out_dir: Path):
        super().__init__(in_dir, out_dir)

    def pdf2xml(self, process_xml: bool = False):
        pattern = re.compile(r"-</")
        super().pdf2xml(pattern, process_xml)

    def xml2yaml(self):
        for f in os.listdir(self.in_dir):
            if f.endswith(".xml"):
                fin_path = os.path.join(self.in_dir, f)
                fout_path = os.path.join(
                    self.out_dir, f.replace(".xml", "_speaker.yaml")
                )
                if not os.path.exists(fout_path):
                    try:
                        xml_parser = XMLParser(fin_path)
                        xml_parser.parse_xml()
                        xml_parser.output_lines(out_path=self.out_dir)
                    except XMLSyntaxError:
                        logging.warning(f"Empty document: {fin_path}")
                        continue

    def pdf2yaml(self, process_xml: bool = False):
        tmp_dir = Path(os.path.commonpath([self.in_dir, self.out_dir])).joinpath("tmp")
        Path.mkdir(tmp_dir, exist_ok=True)
        tmp_out_dir = self.out_dir
        self.out_dir = tmp_dir
        self.pdf2xml(process_xml=process_xml)
        self.in_dir = tmp_dir
        self.out_dir = tmp_out_dir
        self.xml2yaml()
        shutil.rmtree(tmp_dir)
