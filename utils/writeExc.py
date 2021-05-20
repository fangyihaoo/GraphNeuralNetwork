import pandas as pd
import os.path as osp
from openpyxl import load_workbook


def write_excel(dat, opt, filepath = 'result.xlsx') -> None:

    '''
    Save all the output from different schemes into one excel files with different sheets
    '''
    sheetname = f'{opt.data}|{opt.split}'
    layers = opt.layer

    if opt.lamb:
        columnname = [f'{opt.norm} Sparsity {opt.lamb}']
    elif opt.rate:
        columnname = [f'Dropout Rate {opt.rate}']
    else:
        columnname = [f'None']

    # Create a new excel file
    if not osp.exists(filepath): 

        indexnames = [f'{x} Layers' for x in [2**i for i in range(1, 6)]]
        df = pd.DataFrame(0, index = indexnames, columns = columnname)
        df.loc[f'{layers} Layers', columnname] = f'{dat[0]} ({dat[1]})'
        writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
        df.to_excel(writer, sheet_name = sheetname)
        writer.save()
        writer.close()

        return None
    
    excel = pd.ExcelFile(filepath)

    # adding new sheet into this file
    if sheetname not in excel.sheet_names:
        indexnames = [f'{x} Layers' for x in [2**i for i in range(1, 6)]]
        df = pd.DataFrame(0, index = indexnames, columns = columnname)
        df.loc[f'{layers} Layers', columnname] = f'{dat[0]} ({dat[1]})'
        book = load_workbook(filepath)
        writer = pd.ExcelWriter(filepath, engine = 'openpyxl')
        writer.book = book
        df.to_excel(writer, sheet_name = sheetname)
        writer.save()
        writer.close()

        return None

    # updating sheet into this file.
    df = excel.parse(sheetname, index_col=0)
    df.loc[f'{layers} Layers', columnname] = f'{dat[0]} ({dat[1]})'

    book = load_workbook(filepath)
    writer = pd.ExcelWriter(filepath, engine = 'openpyxl')
    writer.book = book
    writer.book.remove(writer.book[sheetname])
    df.to_excel(writer, sheet_name = sheetname)
    writer.save()
    writer.close()

    return None
