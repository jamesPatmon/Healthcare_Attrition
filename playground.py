import pandas as pd

x = pd.Series( ['Geeks', 'for', 'Geeks'] )
df = x.to_frame(name='Test').reset_index().rename(columns={'index': 'Row_Num'})
print(df)
