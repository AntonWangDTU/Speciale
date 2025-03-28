# Get the system information
sys_info <- Sys.info()

# Check the OS and perform actions based on it
if (grepl("Linux", sys_info["sysname"])) {
  if (grepl("arch", Sys.info()["machine"], ignore.case = TRUE)) {
    print("You are using Arch Linux!")
  } else {
    print("You are using a Linux distribution, but not Arch Linux.")
  }
} else if (sys_info["sysname"] == "Windows") {
  print("You are using Windows!")
} else {
  print("Unknown OS!")
}
