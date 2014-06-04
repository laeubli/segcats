setwd('/Users/sam/Documents/ausbildung/uni/msc_ai/thesis/Code/segcats/test_data/observations/exampleSingleGaussianFixationDuration/P/anaylsis')
data.P = read.csv('all-P.obs')

hist(data.P$fixation_dur)
hist(data.P$fixation_dur[data.P$fixation_dur < 1000])
hist(data.P$fixation_dur[data.P$fixation_dur < 2000])
#hist(data.P$fixation_dur[data.P$fixation_dur < 500])
length(data.P$fixation_dur)
