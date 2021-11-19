import pickle
import pandas as pd
import code

class ReadPickle:
  def __init__(self,input):

      self.input = input

  def fix_df(self,df):

      df = df.assign(class_target=int(1))

      df.to_pickle(os.path.join(os.path.dirname(self.input), 'info_df_new.pickle'))

  def run(self):
      
      df = pd.read_pickle(self.input)
      #file = pickle.load(open(self.input,"rb"))
      code.interact(local=locals())
      with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
         print(df)


if __name__ == "__main__":

  #Command line parsing module
  import argparse

  parser = argparse.ArgumentParser(description='ParserForReadingPickleFile')

  parser.add_argument("--file",dest="input",required=True)

  op  = parser.parse_args()


  picklereader = ReadPickle(op.input)

  picklereader.run()
