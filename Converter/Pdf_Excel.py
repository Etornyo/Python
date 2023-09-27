import tabula
import pandas as pd



# Extract from pdf
def extract_from_pdf(pdf_path):
    try:
        tables = tabula.read_pdf(pdf_path,pages='all',multiple_tables=True)
        return tables

    except Exception as