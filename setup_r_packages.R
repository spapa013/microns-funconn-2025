#!/usr/bin/env Rscript

# Install required R packages for funconnect

# Function to install packages if they're not already installed
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new_packages)) {
    message("Installing packages: ", paste(new_packages, collapse=", "))
    install.packages(new_packages, repos="https://cloud.r-project.org/")
  } else {
    message("All required packages are already installed.")
  }
}

# List of required packages
required_packages <- c(
  "glmmTMB",
  "tidyverse",
  "broom.mixed",
  "emmeans",
  "performance",
  "DHARMa"
)

# Install required packages
install_if_missing(required_packages)

# Check if all packages can be loaded
message("Checking if all packages can be loaded...")
for(pkg in required_packages) {
  tryCatch({
    library(pkg, character.only = TRUE)
    message(paste0("Package '", pkg, "' loaded successfully"))
  }, error = function(e) {
    message(paste0("Error loading package '", pkg, "': ", e$message))
  })
}

message("R package setup completed.")
