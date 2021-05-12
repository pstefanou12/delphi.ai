library(truncreg)

# CONSTANTS
TMP_FILE <- '/home/gridsan/stefanou/truncreg/tmp.csv'
RESULT_FILE <- '/home/gridsan/stefanou/truncreg/result.csv'

# Fetch command line arguments
myArgs <- commandArgs(trailingOnly=TRUE)
C <- as.numeric(myArgs[1]) # convert truncation parameter to numeric
dir <- myArgs[2]  # truncation direction

# read in truncated data from the csv file
d <- read.csv(TMP_FILE, header=TRUE, col.names = c("NONE", "X0", "X1"))

X <- as.matrix(d$X0)
y <- as.matrix(d$X1)
df <- data.frame(X=X, y=y)

# truncated regression procedure
trunc_reg <- truncreg(df$y ~ X, data=df, point=C, direction=dir, scaled=TRUE)

# return model coefficients
coef_df <- coef(trunc_reg)

# write results to csv
write.csv(coef_df, RESULT_FILE)
