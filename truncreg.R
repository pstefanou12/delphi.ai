library(truncreg)

# CONSTANTS
TMP_FILE <- '/tmp.csv'
RESULT_FILE <- '/result.csv'

# Fetch command line arguments
myArgs <- commandArgs(trailingOnly=TRUE)
C <- as.numeric(myArgs[1]) # convert truncation parameter to numeric
dir <- myArgs[2]  # truncation direction
out_dir <- myArgs[3]

# read in truncated data from the csv file
d <- read.csv(paste(out_dir, TMP_FILE, sep=''), header=TRUE, col.names = c("NONE", "X0", "X1"))
X <- as.matrix(d$X0)
y <- as.matrix(d$X1)
df <- data.frame(X=X, y=y)
# truncated regression procedure
trunc_reg <- truncreg(df$y ~ X, data=df, point=C, direction=dir, scaled=TRUE)
# return model coefficients
coef_df <- coef(trunc_reg)

# write results to csv
write.csv(coef_df, paste(out_dir, RESULT_FILE, sep=''))
