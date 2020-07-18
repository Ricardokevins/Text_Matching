

filename="vocab.txt"
with open(filename, 'w') as file_object:
    for i in range(52000):
        file_object.write(str(i) + "\n")

    file_object.write("[CLS]" + "\n")
    file_object.write("[ESP]" + "\n")
    file_object.write("[MASK]")
