source("run_methods.R")
source("helper_match_evaluate_multiple.R")
library(flowAssist)
library(tidyr)
library(dplyr)


process_output <- function(output, sample, repetition, path_to_input, y){
  ############ save results to csv's ########
  method <- output[[3]]
  
  # vars for saving
  sample <- gsub(".csv", "", sample)  # drop .csv from sample name.
  save_path <- gsub("input_data", "output", path_to_input)
  
  # if error, catch any errors, but only where prediction isn't NA as that is legit value
  if (output[[1]] == "ERROR" & !is.na(output[[1]])) {
    # if there was an error with the method, just print the error message to the time taken output files so gets picked up when files are checked
    error_message <- paste(deparse(output[[2]][[2]]), as.character(output[[2]][[1]]), sep=" | ")
    cat(paste("ERROR", error_message, sep = "-"), file = paste0(save_path, "/", "time", "_", method, "_", sample, "_", as.character(repetition), ".txt"))
    
  } else {
    # if not error, continue normally.
    
    # write time to file
    time_taken <- output[2][[1]]
    cat(time_taken, file = paste0(save_path, "/", "time", "_", method, "_", sample, "_", as.character(repetition), ".txt"))
    
    # hungarian matching algorithm, match clusters to best fitting labels
    prediction_ints <- unlist(output[1])  # get predictions from clustering algorithm
    
    # convert predictions and labels into the same format.
    # converted unique outputs into consecutive numbers
    r_predictions_ints <- make_readable(prediction_ints, 'predictions')
    
    # encode labels to ints
    r_labels <- recode(unlist(y),
                       "cd4pos"=1,
                       "cd8pos"=2,
                       "dualneg"=3,
                       "dualpos"=4,
                       "gdt"=5,
                       "kappa"=6,
                       "lambd"=7,
                       "nk"=8,
                       "notlymph"=9)
    
    # use hungarian matching algorithm to match predictions to highest f1 labels.
    output <- helper_match_evaluate_multiple(r_predictions_ints, r_labels)
    predictions_to_labels <- output[[5]] # this a named list of the best f1 matching prediction to the corresponding labels.
    ind_F1 <- output[4]  # f1 for each class
    best_F1 <- output[9] # best f1
    n_clus <- output[1] # number of cluster
    
    # convert prediction_ints to the best matching label ints which correspond
    ints_matching_labels <- names(predictions_to_labels)[match(r_predictions_ints, unname(predictions_to_labels))]
    
    # reverse encoding, ints back to labels
    predicted_labels <- recode(ints_matching_labels,
                       "1"="cd4pos",
                       "2"="cd8pos",
                       "3"="dualneg",
                       "4"="dualpos",
                       "5"="gdt",
                       "6"="kappa",
                       "7"="lambd",
                       "8"="nk",
                       "9"="notlymph")
    
    # save predictions to csv
    write.csv(predicted_labels, paste0(save_path, "/", "predictions", "_", method, "_", sample, "_", as.character(repetition), ".csv"),
              row.names = FALSE)
  }
}


args <- commandArgs(trailingOnly=TRUE)
n_repetitions <- as.integer(args[1])
path_to_input <- args[2]

setwd(path_to_input)

all_samples <- list.files()

for (s in 1:length(all_samples)){
  file_name <- all_samples[s]
  
  setwd(path_to_input)  # make sure always go back to correct wd
  samp_df <- read.csv(file_name)
  
  x <- samp_df[,!(names(samp_df) %in% c('label'))]  # drop 'label'
  y <- samp_df['label']
  
  # convert to flowframe to be read by the methods
  ff <- DFtoFF(x)
  
  for(repetition in 1:n_repetitions){
  
    # run methods and time
    methods <- c('cytometree',
                 'flowmeans',
                 'samspectral',
                 'rclusterpp',
                 'flowgrid',
                 'flowsom',
                 'flock'
                 )
    
    for (m in 1:length(methods)){
      method <- methods[m]
      setwd(path_to_input)  # make sure always go back to correct wd
      process_output(try_method(ff, method), file_name, repetition, path_to_input, y)
    }
    
  }
  
}
