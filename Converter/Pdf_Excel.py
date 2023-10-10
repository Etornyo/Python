# import tabula
# import pandas as pd
#
#
#
# # Extract from pdf
# def extract_from_pdf(pdf_path):
#     try:
#         tables = tabula.read_pdf(pdf_path,pages='all',multiple_tables=True)
#         return tables
#
#     except Exception as

import tabula

inp = (r"[pdf name]")
oup = (r"test.cvs")

df = tabula.read_pdf(input_path=inp, pages="all")
tabula.convert_into(input_path = inp, output_path=oup, output_format="cvs", pages="all", stream=True)