import numpy as np

from pdf2image import convert_from_path
from pdfminer.layout import LAParams, LTPage, LTTextBox, LTChar
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator

def load_pdf(pdf_path):
    fp = open(pdf_path, 'rb')
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    pages = PDFPage.get_pages(fp)
    document_text = []
    document_layout = []
    for page in pages:
        interpreter.process_page(page)
        layout = device.get_result()
        page_words = []
        page_boxes = []
        for text_object in layout:
            if isinstance(text_object, LTTextBox):
                for text_line in text_object:
                    word = ""
                    word_box = []
                    for character in text_line:
                        if isinstance(character, LTChar) and character.get_text() != " ":
                            box, text = character.bbox, character.get_text()
                            word += text
                            word_box.append(box) 
                        elif word != "":
                            page_words.append(word)
                            word_box = np.array(word_box)
                            page_boxes.append([np.min(word_box[:, 0]), np.min(word_box[:, 1]), np.max(word_box[:, 2]), np.max(word_box[:, 3])])
                            word = ""
                            word_box = []


                    if word != "":
                        page_words.append(word)
                        word_box = np.array(word_box)
                        page_boxes.append([np.min(word_box[:, 0]), np.min(word_box[:, 1]), np.max(word_box[:, 2]), np.max(word_box[:, 3])])
                            
        page_boxes = [[box[0]/layout.bbox[2], 1 - (box[3]/layout.bbox[3]), box[2]/layout.bbox[2], 1 - (box[1]/layout.bbox[3])] for box in page_boxes]
        document_text.append(page_words)
        document_layout.append(page_boxes)

    fp.close()

    images = convert_from_path(pdf_path)

    for page in range(len(document_layout)):
        boxes = np.array(document_layout[page])
        if np.any(boxes < 0) or np.any(boxes > 1):
            document_layout[page] = list(np.clip(boxes, 0, 1))


    return dict(ocr_tokens=document_text, ocr_boxes=document_layout, images=images)
