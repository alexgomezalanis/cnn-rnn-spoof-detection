import os

def createDirectory(path_location):
  try:
    os.mkdir(path_location)
  except OSError:  
    print ("Creation of the directory %s failed" % path_location)
  else:  
    print ("Successfully created the directory %s " % path_location)