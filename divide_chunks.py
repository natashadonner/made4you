import os
import pandas as pd
import re
from sh import pandoc
from pdf2docx import Converter
df = pd.DataFrame()
root_dir = 'files'
allowed_filetypes = ['.docx', '.pdf']
output = "chunks/output.csv"
docx_file = 'converted/output.docx'


def identify_doc_type(doc):
    '''
    categorizes a plaintext doc based on the format of the toc.
    '''
    if re.search(r'.*\n\n\n-\s{3}.*', doc):
        return "TOC_WITH_TITLE"
    elif re.search(r'-\s{3}.*\n\n.*', doc):
        return "TOC_WITHOUT_TITLE"
    else:
        return "NO_TOC_TITLE"


def read_doc(path):
    '''
    reads a text file and returns toc and full text.
    '''
    doc = str(pandoc(path, "-t", "plain", "--toc", "--standalone"))
    doc_type = identify_doc_type(doc)

    if doc_type == "TOC_WITH_TITLE":
        doc = re.sub('.*\n\n\n-', '-', doc)
        toc, text = doc.split('\n\n', 1)
    elif doc_type == "TOC_WITHOUT_TITLE":
        toc, text = doc.split('\n\n', 1)
    else:
        toc, text = "", doc

    return toc, text


def cleanup_plaintext(text):
    '''
    cleans text
    '''

    text = text.replace("[image]", "").replace("[]", "")

    text = re.sub(r'(?<!\n)\n(?!(\n|-))', ' ', text)

    text = re.sub(r'(:)\s*\n', r'\1 ', text)

    text = re.sub(r'\n{2,}', '\n\n', text)

    text = re.sub(r'(?<!\n) +', ' ', text)
    return text


def split_text(toc, text, min_chunk_size=100):
    '''
    split text into chunks
    '''
    headings = [line.strip('- \n') for line in toc.split('\n') if line.strip()]
    paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]

    current_heading = ""
    text_chunks = []
    buffer = ""
    list_buffer = []

    for para in paragraphs:
        if len(headings) > 0 and para == headings[0]:
            if buffer.strip():
                text_chunks.append(f"{current_heading} [SEP] {buffer}".strip() if current_heading else buffer.strip())
                buffer = ""
            current_heading = headings.pop(0)
            continue

        if para.startswith("-"):
            list_buffer.append(para)
            continue
        elif list_buffer:
            list_chunk = " ".join(list_buffer).strip()
            text_chunks.append(f"{current_heading} [SEP] {list_chunk}".strip() if current_heading else list_chunk)
            list_buffer = []

        if len(buffer) + len(para) < min_chunk_size:
            buffer = f"{buffer} {para}".strip()
        else:
            text_chunks.append(f"{current_heading} [SEP] {buffer}".strip() if current_heading else buffer.strip())
            buffer = para

    if list_buffer:
        list_chunk = " ".join(list_buffer).strip()
        text_chunks.append(f"{current_heading} [SEP] {list_chunk}".strip() if current_heading else list_chunk)

    if buffer.strip():
        text_chunks.append(f"{current_heading} [SEP] {buffer}".strip() if current_heading else buffer.strip())

    return text_chunks


def pdf_converter(full_path):
    cv = Converter(full_path)
    cv.convert(docx_file)
    cv.close()
    return docx_file



for directory, subdirectories, files in os.walk(root_dir):
    for file in files:
        filename, filetype = os.path.splitext(file)
        if filetype in allowed_filetypes:
            full_path = os.path.join(directory, file)
            toc, text = read_doc(full_path)
            print("Toc: " + toc)
            text_cleaned = cleanup_plaintext(text)
            text_chunks = split_text(toc, text_cleaned, 500)
            df_new = pd.DataFrame(text_chunks, columns=["text"])
            df_new[["directory", "filename", "filetype"]] = directory, filename, filetype
            df = pd.concat([df, df_new])
            df.to_csv(output, index=False)

df.reset_index(drop=True, inplace=True)




