TP = 1
TN = 1
FP = 1
FN = 1

print("TP =", TP, "TN =", TN, "FP =", FP, "FN =", FN)

Accuracy = (TP + TN) / (TP + FP + TN + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
f1 = TP * 2 / (TP * 2 + FN + FP)

print("Accuracy =", Accuracy,  "Precision =", Precision, "Recall =", Recall, "f1 =", f1)
