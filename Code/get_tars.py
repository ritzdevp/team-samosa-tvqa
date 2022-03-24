import os

file_ids = open('fileids.txt', 'r')
fid_list = file_ids.readlines()
name_file = open('filenames.txt', 'r')
filename_list = name_file.readlines()

def create_command(FILEID, FILENAME):
  command = """wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/ /p')&id={}" -O {} && rm -rf /tmp/cookies.txt""".format(FILEID, FILEID, FILENAME)
  return command

count = 0
for i in range(len(fid_list)):
  FILENAME = filename_list[i][:-1]
  FILEID = fid_list[i][:-1]
  if (i == len(fid_list) - 1):
    FILENAME = filename_list[i][:]
    FILEID = fid_list[i][:]

  os.system(create_command(FILEID, FILENAME))
  print("$$$$$$$$$$$$$$$$$$")
  print(i, " OF ", len(fid_list), " DONE ")
