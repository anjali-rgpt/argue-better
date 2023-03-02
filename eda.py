import os
import sys

def read_files(path):
    file_contents = []

    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            with open(os.path.join(path, filename), 'r') as f:
                content = f.read()
                file_contents.append(content)

    return file_contents

def main():
    if len(sys.argv) < 2:
        print("Please provide the path to the training dataset folder.")
        return
    path = sys.argv[1]
    
    file_contents = read_files(path)

    print("Number of files read: ", len(file_contents))

if __name__ == "__main__":
    main()