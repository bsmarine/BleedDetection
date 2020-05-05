import pickle
import pandas as pd

class ReadPickle:
  def __init__(self,input):

      self.input = input

  def run(self):
      
      df = pd.read_pickle(self.input)
      #file = pickle.load(open(self.input,"rb"))
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
