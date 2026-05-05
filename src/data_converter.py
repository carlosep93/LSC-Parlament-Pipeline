import logging
import os
import shutil
from pathlib import Path
from typing import Pattern, Union

from lxml.etree import XMLSyntaxError
from scraperwiki import pdftoxml

from io_file import IO
from xml_parser import XMLParser

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)


class DataConverter:
    def __init__(self, in_dir: Path, out_dir: Path):
        self.in_dir = in_dir
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def pdf2xml(self, pattern: Union[Pattern, None], process_xml: bool = False):
        for f in os.listdir(self.in_dir):
            fin_path = os.path.join(self.in_dir, f)
            fout_path = os.path.join(self.out_dir, f.replace(".pdf", ".xml"))
            if not os.path.exists(fout_path):
                file = IO.read_file(fin_path)
                xml_out = pdftoxml(file)
                print('XML_OUT;', xml_out)
                if process_xml:
                    if pattern:
                        xml_out = pattern.sub("</", xml_out)
                IO.write_file(xml_out, fout_path)

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

    def xml2json(self):
        pass
