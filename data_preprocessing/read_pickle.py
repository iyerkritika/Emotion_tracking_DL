import pickle

file2 = open("database.pickle", "rb")
final_matrix, final_label = pickle.load(file2)
print "matrix shape: ", len(final_matrix)
print "label shape: ", len(final_label)
