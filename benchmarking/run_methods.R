library(cytometree)
library(flowMeans)
library(FlowSOM)
library(Rclusterpp)  # needs installing from github install_github("nolanlab/Rclusterpp")
library(SamSPECTRAL)
library(tidyr)

try_method <- function(ff, method){
  # wrapper for try_method, to method to output so doesn't need passing into next function
  # doesn't work in trycatch
  
  out <- try_method_(ff, method)
  out <- append(out, method)
  
  return(out)
}


try_method_ <- function(ff, method) {
  # either return time taken and results of method, or error message
  out <- tryCatch(
    expr = {
      if (method=="cytometree"){
        cytometree(ff)
      } else if (method=="flowmeans"){
        flowmeans(ff)
      } else if (method=="samspectral"){
        samspectral(ff)
      } else if (method=="rclusterpp"){
        rclusterpp(ff)
      } else if (method=="flowgrid"){
        flowgrid(ff)
      } else if (method=="flowsom"){
        flowsom(ff)
      } else if (method=="flock"){
        flock(ff)
      }
    },
    error = function(e){
      out <- list("ERROR", e)  # return the error as the second argument
    }
  )
  return(out)
}


make_readable <-function(old_values, input){
  # take all unique labels and replace with consecutive numbers
  
  # for predictions and labels
  # compared to strings of 'CD4', 'not lymph' etc.
  # whereas some of the methods output predictions as random integers eg. -21, 5, 2, 99 etc.
  
  old_vector <- unlist(old_values)  # to vector
  
  if (input == 'predictions'){
    # unique doesn't like NA's, so replace any NA's with a number that is unique, so that counted incorrect when comes to checking accuracy
    # if leave it in, hungarian algorithm classes it as correct.
    old_vector <- replace_na(old_vector, max(old_vector, na.rm=TRUE)+1)
  }
  
  unique_old <- unique(old_vector)
  unique_new <- seq(1:length(unique_old))
  
  # https://stackoverflow.com/a/16228315/10905324
  new_values <- c(unique_new, old_vector)[match(old_vector, c(unique_old, old_vector))]  # replace all values
  
  new_values_vector <- as.numeric(unlist(new_values))  # to vector again.
  
  return(new_values_vector)
}


cytometree <- function(data){
  start_time <- Sys.time()
  
  Tree <- CytomeTree(data@exprs)
  
  time_taken <- Sys.time() - start_time
  
  return(list(Tree$labels, time_taken))
}


flowgrid <- function(data){
  setwd("/flowgrid")
  
  write.csv(data@exprs, "input_fcs_data.csv", row.names=FALSE)
  
  start_time <- Sys.time()
  
  act_venv_cmd <- file.path("venv", "bin", "activate")
  run_flowgrid_cmd <- "python sample_code.py --f input_fcs_data.csv --n 9  --eps 1.1 --o output_labels.csv"   # n is number of labels
  std_err <- system(paste0(act_venv_cmd, " & ", run_flowgrid_cmd))

  if (std_err != 0){
    # if non zero exit code then print stack trace
    e <- system(paste0(act_venv_cmd, " & ", run_flowgrid_cmd), intern=TRUE)
    stop(toString(e))
  }
  
  time_taken <- Sys.time() - start_time
  
  labels_df <- read.csv("output_labels.csv", header=FALSE)
  list_labels <- labels_df[[1]]
  
  return(list(list_labels, time_taken))
}


flock <- function(data){
  setwd("/flock")
   
  write.table(data@exprs, "input_fcs_data.txt", row.names=FALSE)
  
  start_time <- Sys.time()
  
  run_flock_cmd <- "./flock2 input_fcs_data.txt"
  system(run_flock_cmd)
  
  time_taken <- Sys.time() - start_time
  
  labels_df <- read.table("population_id.txt", header=FALSE)
  list_labels <- labels_df[[1]]
   
  return(list(list_labels, time_taken))
}


flowmeans <- function(data){
  start_time <- Sys.time()
  
  res <- flowMeans(data@exprs, MaxN=9)
  
  time_taken <- Sys.time() - start_time
  
  return(list(res@Label, time_taken))
}


flowsom <- function(data){
  start_time <- Sys.time()
  
  fSOM <- ReadInput(data@exprs, compensate=FALSE, transform=FALSE, scale=FALSE)
  fSOM <- BuildSOM(fSOM, colsToUse = colnames(data))
  fSOM <- BuildMST(fSOM)
  
  metaClustering <- metaClustering_consensus(fSOM$map$codes, k=9)
  metaClustering_perCell <- GetMetaclusters(fSOM, metaClustering)
  
  time_taken <- Sys.time() - start_time
  
  return(list(metaClustering_perCell, time_taken))
}


rclusterpp <- function(data){
  start_time <- Sys.time()
  
  h <- Rclusterpp.hclust(data@exprs)
  
  labels <- cutree(tree=h, k=9)  # define k clusters
  
  time_taken <- Sys.time() - start_time
  
  return(list(labels, time_taken))
}


samspectral <- function(data){
  start_time <- Sys.time()
  
  res <- SamSPECTRAL(data@exprs,
                     normal.sigma=100,
                     separation.factor=1,
                     number.of.clusters=9)  # normal sigma and separation used in weber, no mention in flowcap
  
  time_taken <- Sys.time() - start_time
  
  return(list(res, time_taken))
}