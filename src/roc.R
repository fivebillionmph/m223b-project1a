args = commandArgs(trailingOnly=TRUE)

if(length(args) != 1) {
	stop("need file name")
}

library(ROCR)

f = args[1]

d = read.delim(f)

pred = prediction(d[,2], d[,1])
perf = performance(pred, "tpr", "fpr")
perf.auc = performance(pred, "auc")

print(perf.auc)
plot(perf)
