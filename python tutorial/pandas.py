import pandas as pd
pt = pd.read_excel("./pt.xlsx")
se = pd.read_excel("./SE_V3.xlsx")
total = pd.merge(pt, se, left_on='article', right_on='article')
writer = pd.ExcelWriter('output.xlsx')
total.to_excel(writer)
writer.save()
writer.close()
